import numpy as np
from collections import defaultdict


class VisemeMapper:
    def __init__(self):
        # Phoneme to viseme mapping based on visual similarity
        self.phoneme_to_viseme = {
            # Bilabials - lips pressed together
            'p': 0, 'b': 0, 'm': 0,

            # Labiodentals - lip to teeth
            'f': 1, 'v': 1,

            # Dental/Alveolar - tongue to teeth/alveolar ridge
            't': 2, 'd': 2, 's': 2, 'z': 2, 'th': 2, 'dh': 2, 'n': 2, 'l': 2,

            # Post-alveolar - tongue behind alveolar ridge
            'sh': 3, 'zh': 3, 'ch': 3, 'jh': 3, 'r': 3,

            # Velars - tongue to soft palate
            'k': 4, 'g': 4, 'ng': 4,

            # Open vowels - mouth open
            'aa': 5, 'ae': 5, 'ah': 5, 'ao': 5, 'aw': 5, 'ay': 5,

            # Close vowels - mouth more closed
            'iy': 6, 'ih': 6, 'ey': 6, 'eh': 6,

            # Back vowels - rounded lips
            'uw': 7, 'uh': 7, 'ow': 7, 'oy': 7,

            # Semi-vowels
            'w': 8, 'y': 9,

            # Fricatives
            'h': 10, 'hh': 10, 'hv': 10,

            # Silence
            'sil': 11,

            # Other phonemes mapped to closest viseme
            'dx': 2, 'q': 11, 'em': 0, 'en': 2, 'eng': 4, 'nx': 2,
            'el': 2, 'er': 3, 'ax': 5, 'ix': 6, 'axr': 3, 'ax-h': 5, 'ux': 7
        }

        # Reverse mapping for analysis
        self.viseme_to_phonemes = defaultdict(list)
        for phoneme, viseme in self.phoneme_to_viseme.items():
            self.viseme_to_phonemes[viseme].append(phoneme)

        # Number of visemes
        self.num_visemes = len(set(self.phoneme_to_viseme.values()))

        # Calculate ambiguity scores for each viseme
        self.ambiguity_scores = self._calculate_ambiguity_scores()

        # Create confusion matrix
        self.confusion_matrix = self._create_confusion_matrix()

    def _calculate_ambiguity_scores(self):
        """Calculate ambiguity score for each viseme based on number of phonemes mapped to it"""
        scores = {}
        for viseme_id in range(self.num_visemes):
            num_phonemes = len(self.viseme_to_phonemes[viseme_id])
            # Higher score for more ambiguous visemes
            scores[viseme_id] = min(1.0, num_phonemes / 5.0)
        return scores

    def _create_confusion_matrix(self):
        """Create confusion matrix for phoneme-viseme disambiguation"""
        num_phonemes = len(self.phoneme_to_viseme)
        phoneme_to_idx = {p: i for i, p in enumerate(self.phoneme_to_viseme.keys())}

        # Initialize confusion matrix
        confusion = np.zeros((self.num_visemes, num_phonemes))

        # Fill confusion matrix with probabilities
        for viseme_id in range(self.num_visemes):
            phonemes_in_viseme = self.viseme_to_phonemes[viseme_id]
            prob_per_phoneme = 1.0 / len(phonemes_in_viseme) if phonemes_in_viseme else 0.0

            for phoneme in phonemes_in_viseme:
                phoneme_idx = phoneme_to_idx[phoneme]
                confusion[viseme_id, phoneme_idx] = prob_per_phoneme

        return confusion, phoneme_to_idx

    def get_viseme(self, phoneme):
        """Get viseme ID for a phoneme"""
        return self.phoneme_to_viseme.get(phoneme, 11)  # Default to silence

    def get_ambiguity_score(self, viseme_id):
        """Get ambiguity score for a viseme"""
        return self.ambiguity_scores.get(viseme_id, 0.0)

    def phoneme_to_viseme_sequence(self, phoneme_sequence):
        """Convert phoneme sequence to viseme sequence"""
        return [self.get_viseme(p) for p in phoneme_sequence]

    def get_phoneme_probabilities(self, viseme_id):
        """Get probability distribution over phonemes for a given viseme"""
        confusion_matrix, phoneme_to_idx = self.confusion_matrix
        return confusion_matrix[viseme_id]

    def print_mappings(self):
        """Print viseme mappings for debugging"""
        print("Viseme to Phonemes mapping:")
        for viseme_id in sorted(self.viseme_to_phonemes.keys()):
            phonemes = self.viseme_to_phonemes[viseme_id]
            ambiguity = self.ambiguity_scores[viseme_id]
            print(f"Viseme {viseme_id}: {phonemes} (ambiguity: {ambiguity:.3f})")


# Create global instance
viseme_mapper = VisemeMapper()