"""
Defines the ACR Appropriateness Criteria interface.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import json
import os
from pathlib import Path
from typing import Sequence, Union


class AppropriatenessCriteria:
    def __init__(
        self,
        criteria_fn: Union[Path, str] = os.path.join(
            os.path.dirname(__file__), "ac.json"
        ),
        **kwargs
    ):
        """
        Args:
            criteria_fn: the name of the Appropriateness Criteria file.
        """
        assert os.path.isfile(criteria_fn)
        with open(criteria_fn, "r") as f:
            self._criteria = json.load(f)
        self._topic = list(self._criteria["topics"].keys())
        self._panel = list(self._criteria["panels"].keys())

    def __len__(self) -> int:
        """
        Returns the number of topics in the Appropriateness Criteria dataset.
        Input:
            None.
        Returns:
            The number of topics in the Appropriateness Criteria dataset.
        """
        return len(self.topics)

    @property
    def topics(self) -> Sequence[str]:
        """
        Returns the topics of the Appropriateness Criteria.
        Input:
            None.
        Returns:
            The topics of the Appropriateness Criteria.
        """
        return self._topic

    @property
    def panels(self) -> Sequence[str]:
        """
        Returns the parent panels of the Appropriateness Criteria.
        Input:
            None.
        Returns:
            The parent panels of the Appropriateness Criteria.
        """
        return self._panel

    @property
    def url(self) -> str:
        """
        Returns the source URL of the Appropriateness Criteria.
        Input:
            None.
        Returns:
            The source URL of the Appropriateness Criteria.
        """
        return self._criteria["metadata"]["base_url"]

    @property
    def access_date(self) -> str:
        """
        Returns the date that the Appropriateness Criteria was accessed.
        Input:
            None.
        Returns:
            The date that the Appropriateness Criteria was accessed.
        """
        return self._criteria["metadata"]["access_date"]
