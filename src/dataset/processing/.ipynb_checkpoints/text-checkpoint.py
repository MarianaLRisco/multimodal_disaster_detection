import re


class TextProcessor:

    def __init__(self, lowercase=True, augmenter=None):

        self.lowercase = lowercase
        self.augmenter = augmenter

    def clean_text(self, text):
        text = str(text)

        text = text.encode("utf-8", "ignore").decode("utf-8")
    
        # REMOVE URLs
        text = re.sub(r"http\S+|www\S+", "", text)
    
        # REMOVE @mentions
        text = re.sub(r"@\w+", "", text)
    
        # REMOVE hashtags symbol (optional)
        text = re.sub(r"#", "", text)
    
        # REMOVE extra spaces
        text = re.sub(r"\s+", " ", text).strip()
    
        if self.lowercase:
            text = text.lower()

        return text
        
    def __call__(self, text):
        text_clean = self.clean_text(text)

        if self.augmenter is not None:
            text = self.augmenter(text_clean)

        return text