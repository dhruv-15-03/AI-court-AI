import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from ai_court.utils.nltk_setup import ensure_nltk_resources

# Ensure resources are available when this module is imported
ensure_nltk_resources()

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        # Preserve important legal tokens even if they are stopwords
        self.legal_terms = {
            'plaintiff', 'defendant', 'appeal', 'appeals', 'judgment', 'judgement', 'court', 'section',
            'act', 'article', 'respondent', 'appellant', 'petitioner', 'accused', 'evidence', 'conviction',
            'acquittal', 'forensic', 'witness', 'testimony', 'murder', 'rape', 'ipc', 'crpc', 'dismissed',
            'upheld', 'affirmed', 'ballistic', 'circumstantial', 'alibi', 'bail', 'trial', 'sentence',
            'sentencing', 'compromise', 'settlement', 'damages', 'compensation', 'negligence', 'property'
        }

    def preprocess(self, text: str) -> str:
        """Normalize and lemmatize text while keeping key legal terms."""
        if not isinstance(text, str):
            return ""
        
        # Basic cleanup
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)  # Keep ASCII letters and spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Tokenize
        tokens = word_tokenize(text)
        
        # Keep tokens if either not a stopword or part of a curated legal vocabulary
        filtered = [t for t in tokens if (t not in self.stop_words) or (t in self.legal_terms)]
        
        # Lemmatize (verbs then nouns)
        lemmas = [self.lemmatizer.lemmatize(t, pos='v') for t in filtered]
        lemmas = [self.lemmatizer.lemmatize(t, pos='n') for t in lemmas]
        
        return " ".join(lemmas)

    @staticmethod
    def normalize_outcome(text: str) -> str:
        """Map raw judgment text to a manageable set of outcome classes."""
        t = (text or "").lower()
        # Strong class indicators
        # Acquittal / Conviction overturned
        if any(k in t for k in [
            "acquitted", "acquittal", "conviction overturned", "set aside conviction", "set aside the conviction",
            "set aside", "reversal of conviction", "benefit of doubt", "give benefit of doubt", "acquit the accused",
            "appeal allowed and conviction set aside", "conviction quashed"
        ]):
            return "Acquittal/Conviction Overturned"
        # Conviction upheld / Appeal dismissed
        if any(k in t for k in [
            "appeal dismissed", "appeal is dismissed", "conviction upheld", "conviction affirmed", "convictions upheld",
            "life sentence upheld", "appeal fails", "dismissed on merits", "dismissed in limine", "leave refused",
            "revision dismissed", "petition dismissed as devoid of merit"
        ]):
            return "Conviction Upheld/Appeal Dismissed"
        # Charges / Proceedings quashed
        if any(k in t for k in [
            "charge sheet quashed", "chargesheet quashed", "charge-sheet quashed", "quash the charge sheet",
            "quashing of charge sheet", "charge sheet is quashed", "charge-sheet is quashed"
        ]):
            return "Charge Sheet Quashed"
        if any(k in t for k in [
            "quash", "quashed", "quashing", "fir quashed", "charges quashed", "proceedings quashed",
            "section 482 allowed", "u/s 482 allowed", "proceedings under section 482 are quashed"
        ]):
            return "Charges/Proceedings Quashed"
        # Sentence reduced / modified
        if any(k in t for k in [
            "sentence reduced", "sentence modified", "reduced sentence", "commuted", "converted to",
            "altered sentence", "sentence altered", "imprisonment reduced", "fine reduced"
        ]):
            return "Sentence Reduced/Modified"
        # Bail granted / denied
        if any(k in t for k in [
            "bail granted", "anticipatory bail granted", "interim bail", "enlarged on bail", "released on bail"
        ]):
            return "Bail Granted"
        if any(k in t for k in ["bail denied", "bail rejected", "bail refused"]):
            return "Bail Denied"
        # Relief granted / convicted
        if any(k in t for k in [
            "petition allowed", "writ allowed", "relief granted", "granted protection", "mandated", "directed",
            "ordered", "convicted", "allowed in part", "partly allowed", "set aside order and remand"
        ]):
            return "Relief Granted/Convicted"
        # Relief denied / dismissed
        if any(k in t for k in [
            "petition dismissed", "relief denied", "dismissed as infructuous", "dismissed on merits", "dismissed",
            "no interference called for", "no merit in the petition"
        ]):
            return "Relief Denied/Dismissed"
        return "Other"
