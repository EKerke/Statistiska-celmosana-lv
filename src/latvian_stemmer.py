"""
    This is the implementation of latvian stemmer LatvianStemmer.

    The implementation has been developed by Rihards Krišlauks https://github.com/rihardsk/LatvianStemmer
    And was adapted by E.P.Ķerķe for the batchelor thesis "Statistical stemming for the latvian language".
    
""" 

__author__ = 'rihards'
import fileinput

class Affix:
    def __init__(self, affix, vc, palatalizes):
        self.affix = affix
        self.vc = vc
        self.palatalizes = palatalizes

affixes = [
    Affix("ajiem", 3, False),
    Affix("ajai", 3, False),
    Affix("ajam", 2, False),
    Affix("ajām", 2, False),
    Affix("ajos", 2, False),
    Affix("ajās", 2, False),
    Affix("iem", 2, True),
    Affix("ajā", 2, False),
    Affix("ais", 2, False),
    Affix("ai", 2, False),
    Affix("ei", 2, False),
    Affix("ām", 1, False),
    Affix("am", 1, False),
    Affix("ēm", 1, False),
    Affix("īm", 1, False),
    Affix("im", 1, False),
    Affix("um", 1, False),
    Affix("us", 1, True),
    Affix("as", 1, False),
    Affix("ās", 1, False),
    Affix("es", 1, False),
    Affix("os", 1, True),
    Affix("ij", 1, False),
    Affix("īs", 1, False),
    Affix("ēs", 1, False),
    Affix("is", 1, False),
    Affix("ie", 1, False),
    Affix("u", 1, True),
    Affix("a", 1, True),
    Affix("i", 1, True),
    Affix("e", 1, False),
    Affix("ā", 1, False),
    Affix("ē", 1, False),
    Affix("ī", 1, False),
    Affix("ū", 1, False),
    Affix("o", 1, False),
    Affix("s", 0, False),
    Affix("š", 0, False)
]

def un_palatalize(s, length):
    if s[length] == 'u':
        if endswith(s, length, "kš"):
            length += 1
            s[length - 2] = 's'
            s[length - 1] = 't'
            return s[:length]
        elif endswith(s, length, "ņņ"):
            s[length - 2] = 'n'
            s[length - 1] = 'n'
            return s[:length]

    if endswith(s, length, "pj")\
            or endswith(s, length, "bj")\
            or endswith(s, length, "mj")\
            or endswith(s, length, "vj"):
        length = length - 1
    elif endswith(s, length, "šņ"):
        s[length - 2] = 's'
        s[length - 1] = 'n'
    elif endswith(s, length, "žņ"):
        s[length - 2] = 'z'
        s[length - 1] = 'n'
    elif endswith(s, length, "šļ"):
        s[length - 2] = 's'
        s[length - 1] = 'l'
    elif endswith(s, length, "žļ"):
        s[length - 2] = 'z'
        s[length - 1] = 'l'
    elif endswith(s, length, "ļņ"):
        s[length - 2] = 'l'
        s[length - 1] = 'n'
    elif endswith(s, length, "ļļ"):
        s[length - 2] = 'l'
        s[length - 1] = 'l'
    elif s[length - 1] == 'č':
        s[length - 1] = 'c'
    elif s[length - 1] == 'ļ':
        s[length - 1] = 'l'
    elif s[length - 1] == 'ņ':
        s[length - 1] = 'n'

    return s[:length]


def endswith(s, length, suffix):
    return "".join(s[:length]).endswith(suffix)


def num_vowels(s):
    vowels = {}.fromkeys('aāeēiīouūAĀEĒIĪOUŪ')
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    return count


def stem(s):
    s = list(s)
    numvowels = num_vowels(s)
    length = len(s)

    for affix in affixes:
        if numvowels > affix.vc and length >= len(affix.affix) + 3 and endswith(s, length, affix.affix):
            length -= len(affix.affix)
            s = un_palatalize(s, length) if affix.palatalizes else s[:length]
            break
    return ''.join(s)


def main():
    for line in fileinput.input():
        stems = map(stem, line.rstrip().split())
        print(' '.join(stems))


if __name__ == "__main__":
    main()

from typing import List

def stem_tokens(tokens: List[str]) -> List[str]:
    return [stem(t.lower()) for t in tokens]
