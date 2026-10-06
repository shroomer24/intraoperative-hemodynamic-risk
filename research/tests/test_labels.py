import unittest

import numpy as np

from intraop.data.labels import (
    detect_episodes,
    label_anchors,
    mark_evaluable_episodes,
    matched_episode,
)
from tests.helpers import processed


class LabelTests(unittest.TestCase):
    def at(self, case, time=300):
        anchors = label_anchors(case, detect_episodes(case))
        return anchors.loc[anchors.anchor_time_seconds == time].iloc[0]

    def test_exactly_sixty_low_seconds_confirm_and_fifty_nine_do_not(self):
        for duration, expected in ((59, 0), (60, 1)):
            values = np.full(800, 80.0)
            values[400 : 400 + duration] = 64
            episodes = detect_episodes(processed(values))
            self.assertEqual(len(episodes), expected)
            if expected:
                self.assertEqual(episodes.iloc[0].onset_time_seconds, 400)
                self.assertEqual(episodes.iloc[0].confirmation_time_seconds, 459)

    def test_sixty_five_is_not_low_and_seventy_five_is_eligible(self):
        self.assertEqual(len(detect_episodes(processed(np.full(800, 65)))), 0)
        for value in (65, 70, 75):
            anchor = self.at(processed(np.full(800, value)))
            self.assertEqual(anchor.status, "eligible")
            self.assertEqual(anchor.label, 0)

    def test_ninety_percent_history_coverage_boundary(self):
        for missing, expected in ((30, "eligible"), (31, "ineligible")):
            values = np.full(800, 80.0)
            values[1 : 1 + missing] = np.nan
            self.assertEqual(self.at(processed(values)).status, expected)

    def test_currently_nonhypotensive_rule_includes_anchor_and_exact_sixty_seconds(self):
        for position in (241, 300):
            values = np.full(800, 80.0)
            values[position] = 64
            self.assertEqual(self.at(processed(values)).status, "ineligible")
        values = np.full(800, 80.0)
        values[240] = 64
        self.assertEqual(self.at(processed(values)).status, "eligible")

    def test_missing_recent_second_is_ineligible(self):
        values = np.full(800, 80.0)
        values[300] = np.nan
        self.assertEqual(self.at(processed(values)).status, "ineligible")

    def test_onset_exactly_at_anchor_is_not_a_forecast(self):
        values = np.full(900, 80.0)
        values[300:360] = 64
        episodes = detect_episodes(processed(values))
        self.assertIsNone(matched_episode(episodes, 300))
        self.assertEqual(self.at(processed(values)).status, "ineligible")

    def test_horizon_boundary_is_inclusive_and_next_second_excluded(self):
        for onset, expected in ((301, 1), (600, 1), (601, 0)):
            values = np.full(900, 80.0)
            values[onset : onset + 60] = 64
            anchor = self.at(processed(values))
            self.assertEqual(anchor.status, "eligible")
            self.assertEqual(anchor.label, expected)

    def test_future_through_360_is_required_for_positive_and_negative(self):
        for positive in (False, True):
            values = np.full(900, 80.0)
            if positive:
                values[600:660] = 64
            values[660] = np.nan
            anchor = self.at(processed(values))
            self.assertEqual(anchor.status, "censored")
            self.assertEqual(anchor.reason, "missing_future_map")
            self.assertTrue(np.isnan(anchor.label))

    def test_missing_future_cannot_become_negative(self):
        values = np.full(900, 80.0)
        values[400] = np.nan
        self.assertEqual(self.at(processed(values)).status, "censored")

    def test_full_confirmation_and_surgery_end_at_future_boundary(self):
        values = np.full(661, 80.0)
        values[600:660] = 64
        self.assertEqual(self.at(processed(values)).label, 1)
        self.assertEqual(self.at(processed(values, opend=659)).status, "censored")

    def test_rearming_requires_sixty_valid_recovery_seconds(self):
        for recovery, count in ((59, 1), (60, 2)):
            values = np.full(1000, 80.0)
            values[400:460] = 64
            values[460 + recovery : 520 + recovery] = 64
            episodes = detect_episodes(processed(values))
            self.assertEqual(len(episodes), count)
            if count == 2:
                self.assertFalse(episodes.iloc[0].is_recurrent)
                self.assertTrue(episodes.iloc[1].is_recurrent)

    def test_missingness_breaks_confirmation_and_recovery(self):
        values = np.full(1000, 80.0)
        values[400:460] = 64
        values[520:580] = 64
        values[490] = np.nan
        self.assertEqual(len(detect_episodes(processed(values))), 1)
        values[490] = 80
        values[430] = np.nan
        episodes = detect_episodes(processed(values))
        self.assertEqual(episodes.onset_time_seconds.tolist(), [520])

    def test_first_event_at_surgery_start_has_no_forecast_opportunity(self):
        values = np.full(900, 80.0)
        values[100:160] = 64
        case = processed(values, opstart=100)
        episodes = mark_evaluable_episodes(
            detect_episodes(case), label_anchors(case, detect_episodes(case))
        )
        self.assertEqual(episodes.onset_time_seconds.tolist(), [100])
        self.assertFalse(episodes.iloc[0].evaluable)

    def test_episode_state_resets_at_operation_boundary_for_repeated_subject(self):
        values = np.full(61, 64.0)
        first = detect_episodes(processed(values, case_id="c1", subject_id="s1"))
        second = detect_episodes(processed(values, case_id="c2", subject_id="s1"))
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertFalse(second.iloc[0].is_recurrent)

    def test_anchors_are_recording_aligned_and_history_within_surgery(self):
        case = processed(np.full(1000, 80), opstart=101.5)
        anchors = label_anchors(case, detect_episodes(case))
        self.assertEqual(anchors.iloc[0].anchor_time_seconds, 420)
        self.assertTrue((anchors.history_start_seconds >= case.opstart_seconds).all())
        self.assertTrue((anchors.anchor_time_seconds % 60 == 0).all())
