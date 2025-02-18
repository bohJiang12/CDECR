"""
This module implements a data representation for a single sample within dataset.

A single sample consists of a pair of events from two sentences along with a label,
that is,

Sample := (Sentence_1/Event_1, Sentence_2/Event_2, label)

For an event, it consists of below parts:
    * trigger
    * 2 participants
    * time
    * location
"""

from typing import List
from enum import Enum


DEFAULT_PARTS = 11
NULL_VAL = '<*>'


class EventPair:
    """A pair of `Event` objects"""
    def __init__(self, raw_data: str, is_test_set: bool):
        chunks = raw_data.strip().split('\t')
        if is_test_set:
            chunks = chunks[2:]

        mid = len(chunks) // 2

        self._label = int(chunks[-1])
        self._event_1 = Event(chunks[:mid])
        self._event_2 = Event(chunks[mid:-1])

    @property
    def label(self):
        return self._label

    @property
    def events(self):
        return self._event_1.feature, self._event_2.feature


class TokensMap(Enum):
    """
    A map between ingredients of an event and their
    corresponding indices in raw data
    """
    trigger = (1, 3)
    pp_1 = (3, 5)
    pp_2 = (5, 7)
    time = (7, 9)
    loc = (9, 11)


class Event:
    def __init__(self, parts: List[str]):
        """
        Take a list of constituents of an event, construct an `Event` object
        """
        assert len(parts) == DEFAULT_PARTS
        self._parts = parts

        self._sentence = parts[0]
        self._tokens = self._sentence.strip().split()

        for part in (TokensMap):
            self._add_attrs(part)

        self._feature = self._featurize()

    def _add_attrs(self, part: TokensMap):
        """
        Extract tokens from `self._sentence` attribute given an interval

        ['I love pizza .'] and indices (1, 3), return ['love', 'pizza', '.']

        :params: indices_in_parts: a tuple of start and end indices of an interval
        :return: a list of tokens
        """

        # first, retrieve indices of tokens consisting of a part (e.g. trigger)
        name_in_part, indices_in_part = part.name, part.value
        span_indices = self._parts[slice(*indices_in_part)]
        span_indices = [int(i) for i in span_indices]

        # case: -1 denotes the data contains no information about current part
        #       e.g. no trigger (verb phrase) in the sentence
        if -1 in span_indices:
            self.__setattr__(f"_{name_in_part}", NULL_VAL)
        else:
            s, e = span_indices

            if s == e:
                self.__setattr__(f"_{name_in_part}", self._tokens[s])
            else:
                self.__setattr__(f"_{name_in_part}", ' '.join(self._tokens[s : e]))

    def _featurize(self) -> str:
        """
        Featurize an event into format:

        [EVENT] = [PP_1] [TRIGGER] [PP_2] [TIME] [LOC]
        """

        concat_result = [self.pp1] + [self.trigger] + [self.pp2] + [self.time] + [self.location]

        feature_list = [item for item in concat_result if item != NULL_VAL]

        # return ' '.join(feature_list)
        try:
            return ' '.join(feature_list)
        except TypeError:
            print(feature_list)

    @property
    def sentence(self):
        return self._sentence

    @property
    def trigger(self):
        return self._trigger

    @property
    def pp1(self):
        return self._pp_1

    @property
    def pp2(self):
        return self._pp_2

    @property
    def time(self):
        return self._time

    @property
    def location(self):
        return self._loc

    @property
    def feature(self):
        return self._feature

