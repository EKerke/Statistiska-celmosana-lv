"""
    This is the implementation of latvian stemmer StemmerLV. 

    The java implementation has been developed by Gints Jēkabsons http://www.cs.rtu.lv/jekabsons/nlp.html
    And was translated to python by E.P.Ķerķe for the batchelor thesis "Statistical stemming for the latvian language".
    
""" 
import io
import importlib.resources
from collections import defaultdict
from typing import Dict, List, Optional, Pattern, Set

import regex as re

class Word:
    def __init__(self, word: str, attributes: Optional[str], descr: Optional[str]):
        self.word = word
        self.descr = descr
        self.negatable = False
        if attributes:
            if "-" in attributes:
                attributes = attributes.replace("-", "")
                self.negatable = True
            self.attributes = list(attributes)
        else:
            self.attributes = []

class WordAttrs:
    def __init__(self, attributes: Optional[str], descr: Optional[str], remove_prefix_length: int):
        self.descr = descr
        self.remove_prefix_length = remove_prefix_length
        self.attributes = list(attributes) if attributes else []

class AffixRule:
    def __init__(
        self,
        affix_id: str,
        is_suffix: bool,
        from_str: str,
        to: str,
        condition: str,
        descr: Optional[str],
        further_rules: Dict[str, List["AffixRule"]],
        circumfix_id: str,
        need_affix_id: str,
    ):
        self.affix_id = affix_id
        self.is_suffix = is_suffix
        self.from_str = "" if from_str in ("0", ".") else from_str

        parts = to.split("/")
        raw_to = parts[0]
        self.to = "" if raw_to in ("0", ".") else raw_to

        self.circumfix = False
        self.need_affix = False
        self.further_attributes: List[str] = []

        if len(parts) > 1:
            flags = parts[1]
            if circumfix_id and circumfix_id in flags:
                flags = flags.replace(circumfix_id, "")
                self.circumfix = True
            if need_affix_id and need_affix_id in flags:
                flags = flags.replace(need_affix_id, "")
                self.need_affix = True
            self.further_attributes = list(flags)
            for attr in self.further_attributes:
                further_rules[attr].append(self)

        if condition == ".":
            self.condition = ""
            self.condition_pattern = None
        else:
            cleaned = re.sub(r"(.*?)(\[(.)\])(.*)", r"\1\3\4", condition)
            if "[" in cleaned:
                pattern = f".*{cleaned}" if self.is_suffix else f"{cleaned}.*"
                self.condition_pattern = re.compile(pattern)
                self.condition = ""
            else:
                self.condition = cleaned
                self.condition_pattern = None

        self.descr = descr


class StemmerLV:
    processable_pattern: Pattern = re.compile(r"^[\p{L}\d\-]*[aābcčdeēfgģhiījkķlļmnņoprsštuūvzž]{3}$")
    replace_vowels_pattern: Pattern = re.compile(r"^(.{3}.*?)([aeiouāēīū]{1,3})$")

    def __init__(
        self,
        casefold: bool = True,
        affix_pkg: str = "resources",
        affix_file: str = "lv_LV.aff",
        dic_file: str = "lv_LV.dic",
    ):
        self.casefold = casefold

        self.words: List[Word] = []
        self.words_to_attrs: Dict[str, List[WordAttrs]] = defaultdict(list)

        self.affix_rules_by_id: Dict[str, List[AffixRule]] = {}
        self.suffix_rules: Dict[str, List[AffixRule]] = defaultdict(list)
        self.prefix_rules: Dict[str, List[AffixRule]] = defaultdict(list)
        self.further_rules: Dict[str, List[AffixRule]] = defaultdict(list)

        self.max_suffix_length = 0
        self.min_prefix_length = float("inf")
        self.max_prefix_length = 0

        self.circumfix_id = ""
        self.need_affix_id = ""

        affix_text = self._read_resource(affix_pkg, affix_file)
        dic_text = self._read_resource(affix_pkg, dic_file)
        self.read_affix_rules(affix_text)
        self.read_dictionary(dic_text)

    @staticmethod
    def _read_resource(pkg: str, filename: str) -> str:
        try:
            return importlib.resources.read_text(pkg, filename)
        except (FileNotFoundError, ModuleNotFoundError):
            with io.open(filename, encoding="utf-8") as file:
                return file.read()

    @staticmethod
    def join(*parts: str) -> str:
        return " ".join(part for part in parts if part)

    def read_affix_rules(self, text: str) -> None:
        lines = [line.strip() for line in text.splitlines()]
        affix_id = ""
        rules_list: List[AffixRule] = []

        for line in lines:
            if line.startswith("CIRCUMFIX"):
                self.circumfix_id = line.split()[1]
                continue
            if line.startswith("NEEDAFFIX"):
                self.need_affix_id = line.split()[1]
                continue
            if not (line.startswith("SFX") or line.startswith("PFX")):
                continue

            parts = line.split()
            if parts[1] != affix_id:
                if affix_id and rules_list:
                    self.affix_rules_by_id[affix_id] = rules_list
                affix_id = parts[1]
                rules_list = []

            is_suffix = line.startswith("SFX")
            frm, to = parts[2], parts[3]
            condition = parts[4] if len(parts) >= 5 else "."
            descr = " ".join(parts[5:]) if len(parts) >= 6 else None

            rule = AffixRule(
                affix_id,
                is_suffix,
                frm,
                to,
                condition,
                descr,
                self.further_rules,
                self.circumfix_id,
                self.need_affix_id,
            )
            rules_list.append(rule)

            key = rule.to
            if is_suffix:
                self.suffix_rules[key].append(rule)
                self.max_suffix_length = max(self.max_suffix_length, len(key))
            else:
                self.prefix_rules[key].append(rule)
                self.min_prefix_length = min(self.min_prefix_length, len(key))
                self.max_prefix_length = max(self.max_prefix_length, len(key))

        if affix_id and rules_list:
            self.affix_rules_by_id[affix_id] = rules_list

    def read_dictionary(self, text: str) -> None:
        lines = text.strip().splitlines()[1:]
        for line in lines:
            parts = line.split()
            base, attrs = (parts[0].split("/", 1) + [None])[:2]
            word_str = base.lower() if self.casefold else base
            descr = " ".join(parts[1:]) if len(parts) > 1 else None

            word = Word(word_str, attrs, descr)
            self.words.append(word)

            wa = WordAttrs(attrs, descr, 0)
            self.words_to_attrs[word_str].append(wa)

            for attr in wa.attributes:
                for rule in self.affix_rules_by_id.get(attr, []):
                    if not rule.is_suffix:
                        pref = rule.to + word_str
                        self.words_to_attrs[pref].append(WordAttrs(attrs, descr, len(rule.to)))

    def stem(self, word: str, stop_on_first: bool = False, allow_guess: bool = True) -> str:
        w = word.lower() if self.casefold else word
        if len(w) <= 1 or not self.processable_pattern.match(w):
            return word

        lemmas = self.lemmatize(w, stop_on_first, False, allow_guess)
        if not lemmas:
            return word

        best = None
        for cand in lemmas:
            stemmed = self.stem_lemma(cand)
            if (w.startswith("ne") and not stemmed.startswith("ne")) or (
                w.startswith("visne") and not stemmed.startswith(("visne", "ne"))
            ):
                stemmed = "ne" + stemmed

            better = best is None or len(stemmed) < len(best) or (
                len(stemmed) == len(best) and w.startswith(stemmed)
            )
            if better:
                best = stemmed

        return best

    @classmethod
    def stem_lemma(cls, lemma: str) -> str:
        if len(lemma) <= 3:
            return lemma

        last = lemma[-1]
        if last == "š" and lemma[-2] in ("ņ", "j", "ļ"):
            return lemma[:-1]
        if last == "t" and lemma[-2] in list("ābeēgīklmopsruūz"):
            return lemma[:-1]
        if last == "s":
            if lemma.endswith("ties") and len(lemma) > 6 and lemma[-5:-4] in list("ābeēgīklmopsruūz"):
                return lemma[:-4]
            return lemma[:-1]

        return cls.replace_vowels_pattern.sub(r"\1", lemma)

    def has_bad_ending(self, lemma: str) -> bool:
        bad_endings = [
            "šanaš", "šanāš", "šanuš", "šaneš", "šanš", "ddis", "skss", "aiēt",
            "aāt", "aēt", "āēt", "eēt", "ēēt", "īēt", "aīt", "eīt", "iīt", "oīt", "uīt",
            "sst", "tst", "šst", "jpt", "zpt", "cpt", "dpt", "mst", "glt", "drt", "vst", "ļlt",
            "nbt", "mbt", "npt",
            "sss", "šss", "aas", "mss", "vss", "ļļs", "ļls", "ļļš",
            "aīs", "aās", "eīs", "eās", "iīs", "iās", "uīs", "uās",
            "šuš", "eiš", "čuš", "zdš", "vcš", "ftš", "ntš",
            "ie", "jt", "ct", "ee", "ae", "ff", "fš", "nš", "sš", "šš", "tš", "zš", "žš",
            "čt", "dt", "ft", "ģt", "ht", "jt", "ķt", "ļt", "nt", "ņt", "št", "tt", "vt", "žt"
        ]
        if any(lemma.endswith(be) for be in bad_endings):
            return True
        if (lemma.endswith("ēs") and lemma != "mēs") or (lemma.endswith("ūs") and lemma != "jūs"):
            return True
        return False

    def remove_bad_lemmas(self, lemmas: Set[str], original: str) -> None:
        lemmas.difference_update({l for l in lemmas if l.lower() != original and self.has_bad_ending(l)})

    def _check_rules(
        self,
        word: str,
        rules: List[AffixRule],
        removed_affix: bool,
        removed_prefix: bool,
        info: str,
        lemmas: Set[str],
        guessed: Optional[Set[str]],
        stop_on_first: bool,
        report: bool,
    ) -> bool:
        found = False

        for rule in rules:
            if rule.is_suffix:
                if rule.circumfix and not removed_prefix:
                    continue
                if rule.to and not word.endswith(rule.to):
                    continue
                form = word[: -len(rule.to)] if rule.to else word
                form = form + rule.from_str if rule.from_str else form
                if rule.condition_pattern and not rule.condition_pattern.match(form):
                    continue
                if rule.condition and not form.endswith(rule.condition):
                    continue
            else:
                if rule.to and not word.startswith(rule.to):
                    continue
                form = word[len(rule.to) :] if rule.to else word
                form = rule.from_str + form if rule.from_str else form
                if rule.condition_pattern and not rule.condition_pattern.match(form):
                    continue
                if rule.condition and not form.startswith(rule.condition):
                    continue

            attrs_list = self.words_to_attrs.get(form)
            if attrs_list:
                for wa in attrs_list:
                    if (not rule.need_affix or removed_affix) and rule.affix_id in wa.attributes:
                        lemma = form[wa.remove_prefix_length :] if wa.remove_prefix_length > 0 else form
                        if report:
                            lemmas.add(self.join(lemma, wa.descr or "", rule.descr or info))
                        else:
                            lemmas.add(lemma)
                        if stop_on_first:
                            return True
                        found = True

            further = self.further_rules.get(rule.affix_id)
            if further:
                found |= self._check_rules(
                    form, further, True, not rule.is_suffix, info, lemmas, guessed, stop_on_first, report
                )
            elif guessed is not None and not lemmas:
                guessed.add(form)

        return found

    def lemmatize(self, word: str, stop_on_first: bool, report: bool, allow_guess: bool = False) -> Set[str]:
        w = word.lower() if self.casefold else word
        if len(w) <= 1:
            return set()

        lemmas: Set[str] = set()

        for wa in self.words_to_attrs.get(w, []):
            lemmas.add(w[wa.remove_prefix_length :])
            if stop_on_first:
                return lemmas

        guessed: Optional[Set[str]] = set() if allow_guess else None

        for i in range(min(self.max_suffix_length, len(w)), -1, -1):
            suffix = w[-i:] if i > 0 else ""
            rules = self.suffix_rules.get(suffix)
            if rules and self._check_rules(w, rules, False, False, "", lemmas, guessed, stop_on_first, report):
                if stop_on_first:
                    return lemmas

        for i in range(self.min_prefix_length, min(self.max_prefix_length, len(w)) + 1):
            prefix = w[:i]
            rules = self.prefix_rules.get(prefix)
            if rules and self._check_rules(w, rules, False, False, "", lemmas, guessed, stop_on_first, report):
                if stop_on_first:
                    return lemmas

        if allow_guess and not lemmas and guessed:
            self.remove_bad_lemmas(guessed, w)
            return guessed

        return lemmas

    def list_forms(self, lemma: str, report: bool = False) -> List[str]:
        lemma = lemma.lower() if self.casefold else lemma
        for word in self.words:
            if word.word == lemma:
                forms: List[str] = [self.join(word.word, word.descr or "") if report else word.word]
                self._list_forms(word.word, word.attributes, word.descr, False, forms, report)
                if word.negatable:
                    neg = "ne" + word.word
                    forms.append(self.join(neg, word.descr or "") if report else neg)
                    self._list_forms(neg, word.attributes, word.descr, True, forms, report)
                return forms
        return []

    def _list_forms(
        self,
        word: str,
        attributes: List[str],
        descr: Optional[str],
        negation: bool,
        forms: List[str],
        report: bool,
    ) -> None:
        for attr in attributes:
            for rule in self.affix_rules_by_id.get(attr, []):
                if rule.is_suffix:
                    if rule.condition_pattern and not rule.condition_pattern.match(word):
                        continue
                    base = word[: -len(rule.from_str)] if rule.from_str else word
                    form = base + rule.to if rule.to else base
                    forms.append(self.join(form, descr or "", rule.descr or "") if report else form)
                    if rule.further_attributes:
                        self._list_forms(form, rule.further_attributes, descr, negation, forms, report)
                else:
                    if rule.condition_pattern and not rule.condition_pattern.match(word):
                        continue
                    if rule.to == "jā" and negation:
                        continue
                    base = word[len(rule.from_str) :] if rule.from_str else word
                    form = rule.to + base if rule.to else base
                    if not rule.need_affix:
                        forms.append(self.join(form, descr or "", rule.descr or "") if report else form)
                    if rule.further_attributes:
                        self._list_forms(form, rule.further_attributes, descr, negation, forms, report)

    def list_dictionary_with_all_forms(self, report: bool = False) -> Dict[str, List[str]]:
        return {w.word: self.list_forms(w.word, report) for w in self.words}


_lv = StemmerLV()


def stemlv_token(token: str) -> str:
    """Return the stem of a single word."""
    return _lv.stem(token)


def stemlv_tokens(tokens: List[str]) -> List[str]:
    """Stem every token in a list."""
    return [stemlv_token(t) for t in tokens]