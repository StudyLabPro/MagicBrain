import pytest
from magicbrain.training.data_partitioner import DataPartitioner


SAMPLE_TEXT = "abcdefghijklmnopqrstuvwxyz" * 10  # 260 chars


class TestRoundRobin:
    def test_equal_ish_partitions(self):
        parts = DataPartitioner.round_robin(SAMPLE_TEXT, 4)
        assert len(parts) == 4
        sizes = [len(p) for p in parts]
        assert max(sizes) - min(sizes) <= 1

    def test_all_data_preserved(self):
        parts = DataPartitioner.round_robin(SAMPLE_TEXT, 3)
        # Reconstruct: round-robin means char i goes to partition i%n
        reconstructed = [""] * len(SAMPLE_TEXT)
        cursors = [0] * 3
        for i in range(len(SAMPLE_TEXT)):
            p_idx = i % 3
            reconstructed[i] = parts[p_idx][cursors[p_idx]]
            cursors[p_idx] += 1
        assert "".join(reconstructed) == SAMPLE_TEXT

    def test_single_partition(self):
        parts = DataPartitioner.round_robin(SAMPLE_TEXT, 1)
        assert len(parts) == 1
        assert parts[0] == SAMPLE_TEXT

    def test_empty_data(self):
        parts = DataPartitioner.round_robin("", 3)
        assert parts == ["", "", ""]


class TestOverlapping:
    def test_overlap_region_exists(self):
        parts = DataPartitioner.overlapping(SAMPLE_TEXT, 2, overlap_chars=20)
        assert len(parts) == 2
        # Both parts together should be longer than original due to overlap
        assert len(parts[0]) + len(parts[1]) > len(SAMPLE_TEXT)

    def test_all_chars_covered(self):
        parts = DataPartitioner.overlapping(SAMPLE_TEXT, 3, overlap_chars=10)
        covered = set()
        for part in parts:
            for ch in part:
                covered.add(ch)
        original_chars = set(SAMPLE_TEXT)
        assert covered == original_chars

    def test_single_partition(self):
        parts = DataPartitioner.overlapping(SAMPLE_TEXT, 1)
        assert parts == [SAMPLE_TEXT]

    def test_last_partition_includes_end(self):
        parts = DataPartitioner.overlapping(SAMPLE_TEXT, 3, overlap_chars=5)
        assert parts[-1].endswith(SAMPLE_TEXT[-1])


class TestShufflePartition:
    def test_all_data_preserved(self):
        parts = DataPartitioner.shuffle_partition(SAMPLE_TEXT, 3, chunk_size=10, seed=42)
        reconstructed = "".join(parts)
        assert sorted(reconstructed) == sorted(SAMPLE_TEXT)
        assert len(reconstructed) == len(SAMPLE_TEXT)

    def test_deterministic(self):
        p1 = DataPartitioner.shuffle_partition(SAMPLE_TEXT, 3, chunk_size=10, seed=123)
        p2 = DataPartitioner.shuffle_partition(SAMPLE_TEXT, 3, chunk_size=10, seed=123)
        assert p1 == p2

    def test_different_seeds_differ(self):
        p1 = DataPartitioner.shuffle_partition(SAMPLE_TEXT, 2, chunk_size=10, seed=1)
        p2 = DataPartitioner.shuffle_partition(SAMPLE_TEXT, 2, chunk_size=10, seed=2)
        assert p1 != p2

    def test_correct_partition_count(self):
        parts = DataPartitioner.shuffle_partition(SAMPLE_TEXT, 5, chunk_size=20, seed=0)
        assert len(parts) == 5
