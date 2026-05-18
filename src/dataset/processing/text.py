import re


class TextProcessor:

    def __init__(self, lowercase=True):

        self.lowercase = lowercase

    def clean_text(self, text):

        text = str(text)

        # REMOVE URLS
        text = re.sub(
            r"http\\S+|www\\S+",
            "",
            text
        )

        # REMOVE EXTRA SPACES
        text = re.sub(
            r"\\s+",
            " ",
            text
        ).strip()

        # LOWERCASE
        if self.lowercase:
            text = text.lower()

        return text

    def __call__(self, text):

        return self.clean_text(text)