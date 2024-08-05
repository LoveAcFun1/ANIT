import nltk
from nltk.corpus import brown, twitter_samples

freqs = nltk.FreqDist(w.lower() for w in brown.words())

HIGHEST_WORDS = freqs.most_common(5000)
LOWEST_WODDS = freqs.most_common()[:-5000]
RANDOM_WORDS = freqs.keys()
ICL_DATASET = twitter_samples.strings()