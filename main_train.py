import os
# Suppress SDL3 warnings by disabling SDL audio output
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_ASSERT'] = 'no'  # Optional: disables some SDL runtime checks

import logging
import re
from collections import Counter
import joblib

import mido
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from hmmlearn import hmm

# Configure logging to show progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurable constants
MAX_SAMPLES = 2000
TOP_CHORDS = 30
DURATION_SECS = 60
TEMPO_BPM = 120

# Paths
data_path = 'midicaps.csv'

class CaptionDataset:
    def __init__(self, path, max_samples=None):
        df = pd.read_csv(path)
        if max_samples:
            df = df.head(max_samples)
        if df['caption'].isnull().any():
            raise ValueError("Found null captions in dataset")
        df['chord_list'] = df['all_chords'].str.split('-')
        self.df = df
        logger.info(f"Loaded {len(df)} samples from {path}")

class TextEmbedder:
    def __init__(self, texts, max_features=1000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
        self.vectorizer.fit(texts)
        logger.info('TF-IDF fitted on %d captions', len(texts))

    def embed(self, texts):
        return self.vectorizer.transform(texts).toarray()

    def save(self, path):
        joblib.dump(self.vectorizer, path)

    @staticmethod
    def load(path):
        embedder = TextEmbedder([])
        embedder.vectorizer = joblib.load(path)
        return embedder

class GenreMoodClassifier:
    def __init__(self):
        self.le_genre = LabelEncoder()
        self.le_mood = LabelEncoder()
        self.clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))

    def fit(self, X_text, y_genre, y_mood, embedder):
        # Encode labels
        yg = self.le_genre.fit_transform(y_genre)
        ym = self.le_mood.fit_transform(y_mood)
        # Split into train/test
        X_train, X_test, yg_train, yg_test, ym_train, ym_test = \
            train_test_split(X_text, yg, ym, train_size=0.8, random_state=42)
        # Embed texts
        X_train_emb = embedder.embed(X_train)
        X_test_emb = embedder.embed(X_test)
        # Train classifier
        self.clf.fit(X_train_emb, np.vstack([yg_train, ym_train]).T)
        logger.info('Classifier training complete')
        # Predict on test
        pred = self.clf.predict(X_test_emb)
        # F1 scores
        f1_g = f1_score(yg_test, pred[:,0], average='weighted')
        f1_m = f1_score(ym_test, pred[:,1], average='weighted')
        logger.info(f'F1 Score - Genre: {f1_g:.3f}, Mood: {f1_m:.3f}')
        # Classification reports
        logger.info('Classification report for genre:')
        logger.info('\n' + classification_report(yg_test, pred[:,0], labels=np.unique(yg_test),
                         target_names=self.le_genre.inverse_transform(np.unique(yg_test))))
        logger.info('Classification report for mood:')
        logger.info('\n' + classification_report(ym_test, pred[:,1], labels=np.unique(ym_test),
                         target_names=self.le_mood.inverse_transform(np.unique(ym_test))))

    def predict(self, captions, embedder):
        emb = embedder.embed(captions)
        pg_idx, pm_idx = self.clf.predict(emb)[0]
        return self.le_genre.inverse_transform([pg_idx])[0], self.le_mood.inverse_transform([pm_idx])[0]

    def save(self, path_prefix='model'):
        joblib.dump(self.clf, f'{path_prefix}_clf.pkl')
        joblib.dump(self.le_genre, f'{path_prefix}_genre_encoder.pkl')
        joblib.dump(self.le_mood, f'{path_prefix}_mood_encoder.pkl')

    def load(self, path_prefix='model'):
        self.clf = joblib.load(f'{path_prefix}_clf.pkl')
        self.le_genre = joblib.load(f'{path_prefix}_genre_encoder.pkl')
        self.le_mood = joblib.load(f'{path_prefix}_mood_encoder.pkl')

class ChordHMM:
    def __init__(self, sequences, top_k=TOP_CHORDS):
        counts = Counter(ch for seq in sequences for ch in seq)
        top_chords = {ch for ch, _ in counts.most_common(top_k)}
        self.states = list(sorted(top_chords)) + ['UNK']
        self.state_idx = {s:i for i,s in enumerate(self.states)}
        self.N = len(self.states)
        # Laplace-smoothed counts
        trans = np.ones((self.N, self.N))
        start = np.ones(self.N)
        emit_counts = np.ones((self.N, self.N))
        for seq in sequences:
            reduced = [ch if ch in top_chords else 'UNK' for ch in seq]
            start[self.state_idx[reduced[0]]] += 1
            for a, b in zip(reduced, reduced[1:]):
                i, j = self.state_idx[a], self.state_idx[b]
                trans[i,j] += 1
                emit_counts[i,self.state_idx[b]] += 1
        self.startprob_ = start / start.sum()
        self.transmat_ = trans / trans.sum(axis=1, keepdims=True)
        self.emissionprob_ = emit_counts / emit_counts.sum(axis=1, keepdims=True)
        self.model = hmm.CategoricalHMM(n_components=self.N, init_params='', params='')
        self.model.startprob_ = self.startprob_
        self.model.transmat_ = self.transmat_
        self.model.emissionprob_ = self.emissionprob_
        logger.info('Initialized HMM with %d states', self.N)

    def sample(self, n_steps):
        X, _ = self.model.sample(n_steps)
        return [self.states[i] for i in X.flatten()]

class MidiRenderer:
    NOTE_BASE = {'C':60,'D':62,'E':64,'F':65,'G':67,'A':69,'B':71}
    INTERVALS = {'maj':[0,4,7],'min':[0,3,7]}

    @staticmethod
    def chord_to_notes(chord):
        m = re.match(r"^([A-G])(#|b)?(m|maj|min|dim|aug|sus\d*)?$", chord)
        if not m:
            return [60,64,67]  # Default C major
        root, acc, quality = m.groups()
        pitch = MidiRenderer.NOTE_BASE[root]
        if acc=='#': pitch += 1
        elif acc=='b': pitch -= 1
        intervals = MidiRenderer.INTERVALS.get('min' if quality == 'm' else 'maj', [0,4,7])
        return [pitch+i for i in intervals]

    @staticmethod
    def render(chord_sequence, duration_secs=DURATION_SECS, tempo_bpm=TEMPO_BPM, output='output.mid'):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_bpm)))
        tpq = mid.ticks_per_beat
        for chord in chord_sequence:
            notes = MidiRenderer.chord_to_notes(chord)
            dt = int(tpq * (60/tempo_bpm) * np.random.uniform(0.5,1.5))
            vel = int(np.random.uniform(40,100))
            for n in notes:
                track.append(mido.Message('note_on', note=n, velocity=vel, time=0))
            for n in notes:
                track.append(mido.Message('note_off', note=n, velocity=vel, time=dt if n==notes[0] else 0))
        mid.save(output)
        logger.info('MIDI saved to %s', output)
        return output

if __name__ == '__main__':
    # Load dataset and embedder
    ds = CaptionDataset(data_path, max_samples=MAX_SAMPLES)
    embedder = TextEmbedder(ds.df['caption'].tolist())
    embedder.save('tfidf_vectorizer.pkl')

    # Train Genre-Mood classifier
    gmc = GenreMoodClassifier()
    gmc.fit(ds.df['caption'].tolist(), ds.df['genre'], ds.df['mood'], embedder)
    gmc.save('genre_mood')

    # Train HMM for chord sequence generation
    chord_hmm = ChordHMM(ds.df['chord_list'].tolist())

    # Accept text input for custom caption
    input_caption = input("Enter a text sequence (caption): ")

    # Get genre and mood predictions for the input caption (optional)
    genre, mood = gmc.predict([input_caption], embedder)
    print(f"Predicted Genre: {genre}")
    print(f"Predicted Mood: {mood}")

    # Generate chord sequence based on the input caption
    # You can tune the number of steps for more or less complexity in the sequence
    seq = chord_hmm.sample(int(DURATION_SECS * 2))
    
    # Render MIDI file from chord sequence
    midi_file = MidiRenderer.render(seq)
    print(f"MIDI file saved: {midi_file}")
    print(midi_file)
