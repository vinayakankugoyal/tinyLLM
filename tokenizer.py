from __future__ import annotations
from collections import defaultdict
import pickle

__all__ = ["BPE", "create", "load"]


class _Node:
    def __init__(self, data: str):
        self.token = data
        self.next = None

    def __repr__(self):
        out = []
        current = self
        while current:
            out.append(current.token + "->")
            current = current.next
        out.append("None")
        return "".join(out)

    def pairs(self) -> list[tuple[tuple[str, str], _Node]]:
        result = []

        current = self
        while current is not None and current.next is not None:
            result.append(((current.token, current.next.token), current))
            current = current.next

        return result

    def merge(self):
        self.token = self.token + self.next.token
        self.next = self.next.next


def _text_to_words(text: str) -> list[str]:
    result = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        line_words = line.split()

        if len(line_words) == 0:
            continue

        if i != 0:
            line_words[0] = "\n" + line_words[0]

        for i in range(1, len(line_words)):
            line_words[i] = " " + line_words[i]

        result.extend(line_words)

    return result


def _word_to_linked_list(word: str) -> _Node:
    head = _Node(word[0])
    current = head
    for char in word[1:]:
        current.next = _Node(char)
        current = current.next

    return head


def _words_to_linked_lists(words: list[str]) -> list[_Node]:
    result = []
    for word in words:
        result.append(_word_to_linked_list(word))
    return result


def _bpe(words: list[str], vocab_size: int = 500) -> tuple[tuple[str, str], dict]:
    merge_rules = []

    vocabulary = {char for word in words for char in word}
    vocabulary = list(sorted(vocabulary))

    linked_lists = _words_to_linked_lists(words)

    while len(vocabulary) < vocab_size:
        pair_freqs = defaultdict(list)

        for linked_list in linked_lists:
            pairs = linked_list.pairs()

            for pair, node in pairs:
                pair_freqs[pair].append(node)

        most_frequent_pair = max(pair_freqs, key=lambda k: len(pair_freqs[k]))

        merge_rules.append(most_frequent_pair)
        vocabulary.append(most_frequent_pair[0] + most_frequent_pair[1])

        for node in pair_freqs[most_frequent_pair]:
            node.merge()

    return (merge_rules, vocabulary)


class BPE:
    def __init__(self, merge_rules, vocabulary):
        self.__merge_rules = merge_rules
        self.__vocabulary = vocabulary
        self.__token_to_int = {c: i for i, c in enumerate(self.__vocabulary)}
        self.__int_to_token = {i: c for i, c in enumerate(self.__vocabulary)}

    def __encode_word_list(self, word_list: _Node) -> list[int]:
        current = word_list
        result = []
        while current:
            result.append(self.__token_to_int[current.token])
            current = current.next

        return result

    def encode(self, text: str) -> list[int]:
        words = _text_to_words(text)
        linked_lists = _words_to_linked_lists(words)

        for merge_pair in self.__merge_rules:
            all_pairs = defaultdict(list)
            for linked_list in linked_lists:
                pairs = linked_list.pairs()

                for pair, node in pairs:
                    all_pairs[pair].append(node)

            for node in all_pairs[merge_pair]:
                node.merge()

        result = []
        for linked_list in linked_lists:
            result.extend(self.__encode_word_list(linked_list))

        return result

    def decode(self, encoding: list[int]) -> str:
        result = []
        for encoded in encoding:
            result.append(self.__int_to_token[encoded])

        return "".join(result)

    def save(self, output: str):
        with open(output, "wb") as f:
            pickle.dump(
                {
                    "merge_rules": self.__merge_rules,
                    "vocabulary": self.__vocabulary,
                },
                f,
            )

    def vocab_size(self) -> int:
        return len(self.__vocabulary)


def create(input: str, vocab_size: int = 500) -> BPE:
    with open(input, "r") as f:
        text = f.read()

    words = _text_to_words(text)
    merge_rules, vocabulary = _bpe(words, vocab_size)

    return BPE(merge_rules, vocabulary)


def load(save_file: str) -> BPE:
    with open(save_file, "rb") as f:
        tokenizer_params = pickle.load(f)

        return BPE(tokenizer_params["merge_rules"], tokenizer_params["vocabulary"])
