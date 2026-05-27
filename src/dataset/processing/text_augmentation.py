from deep_translator import GoogleTranslator
import random

class BackTranslationAugmenter:

    def __init__(self, p=0.3):
        self.p = p
        self.en_es = GoogleTranslator(source="en", target="es")
        self.es_en = GoogleTranslator(source="es", target="en")

    def augment(self, text):

        if random.random() > self.p:
            return text

        es = self.en_es.translate(text)
        back = self.es_en.translate(es)

        return back

    def __call__(self, text):
        return self.augment(text)