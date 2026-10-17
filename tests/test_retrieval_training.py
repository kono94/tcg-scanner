import copy
import math
import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from training.card_recognizer.retrieval_training import ArcFace, CardEmbedding, augment_card, build_training_plan, identity_key
from training.card_recognizer.train_retrieval import retrieval_metrics


def record(card, digest, game='onepiece', language='en', source=None):
    row = {'identity': {'game': game, 'language': language, 'card_id': card},
           'image': card + '.png', 'sha256': digest}
    if source:
        row['source_group'] = source
    return row


def fixture():
    return {'gallery': [record('OP01-001', 'a'), record('OP01-001_p1', 'b', language='ja'),
                        record('OP01-002', 'c'), record('OP01-003', 'd'),
                        record('OP01-004', 'd'), record('OP01-005', 'e'),
                        record('OP01-006', 'a')],
            'splits': {'train': {'queries': [record('OP01-002', 'q1', source='train-video')]},
                       'val': {'queries': [record('OP01-001', 'q2', source='val-video')]},
                       'test': {'queries': [record('OP01-005', 'q3', source='test-video')]}}}


class RetrievalTrainingTests(unittest.TestCase):
    def test_holdouts_exclude_languages_variants_and_byte_aliases_before_sampling(self):
        manifest = fixture()
        plan = build_training_plan(manifest, max_classes=10)
        ids = {i for g in plan['classes'] for i in g['identities']}
        self.assertEqual(ids, {'onepiece:en:OP01-002', 'onepiece:en:OP01-003', 'onepiece:en:OP01-004'})
        self.assertEqual(plan['training_class_count'], 2)  # identical fronts share one proxy
        self.assertEqual(plan['training_real_query_count'], 1)
        self.assertIn('onepiece:ja:OP01-001_p1', plan['excluded_training_reference_identities'])
        self.assertIn('onepiece:en:OP01-006', plan['excluded_training_reference_identities'])
        self.assertEqual(plan, build_training_plan(manifest, max_classes=10))

    def test_same_video_or_query_bytes_cannot_cross_splits(self):
        for field, value in [('source_group', 'train-video'), ('sha256', 'q1')]:
            manifest = copy.deepcopy(fixture())
            manifest['splits']['test']['queries'][0][field] = value
            with self.assertRaises(ValueError):
                build_training_plan(manifest)

    def test_known_ambiguous_reference_and_its_alias_are_excluded(self):
        plan = build_training_plan(fixture(), excluded_identities=('onepiece:en:OP01-003',))
        ids = {i for g in plan['classes'] for i in g['identities']}
        self.assertNotIn('onepiece:en:OP01-003', ids)
        self.assertNotIn('onepiece:en:OP01-004', ids)

    def test_warmup_freezes_backbone_then_only_last_blocks_receive_updates(self):
        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=4)
                self.stem = torch.nn.Linear(3, 4)
                self.encoder = torch.nn.Module()
                self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(4)])
                self.layernorm = torch.nn.LayerNorm(4)

            def forward(self, pixel_values):
                x = self.stem(pixel_values.mean(dim=(-1, -2)))
                for layer in self.encoder.layer:
                    x = torch.tanh(layer(x))
                return SimpleNamespace(last_hidden_state=self.layernorm(x)[:, None, :])

        encoder = CardEmbedding(TinyBackbone(), dimension=4)
        encoder.configure_backbone(0)
        self.assertFalse(any(p.requires_grad for p in encoder.backbone.parameters()))
        encoder.configure_backbone(2)
        optimizer = torch.optim.SGD([p for p in encoder.parameters() if p.requires_grad], lr=.01)
        before = encoder.backbone.encoder.layer[-1].weight.detach().clone()
        x = torch.randn(2, 3, 224, 224)
        encoder(x)[:, 0].sum().backward(); optimizer.step()
        self.assertIsNone(encoder.backbone.encoder.layer[0].weight.grad)
        self.assertIsNotNone(encoder.backbone.encoder.layer[-1].weight.grad)
        self.assertFalse(torch.equal(before, encoder.backbone.encoder.layer[-1].weight))

    def test_arcface_matches_angular_formula_and_has_finite_gradients(self):
        head = ArcFace(2, 2, margin=.2, scale=16)
        with torch.no_grad():
            head.weight.copy_(torch.eye(2))
        vector = torch.tensor([[.8, .6]], requires_grad=True)
        logits = head(vector, torch.tensor([0]))
        self.assertAlmostEqual(logits[0, 0].item(), math.cos(math.acos(.8) + .2) * 16, places=5)
        self.assertAlmostEqual(logits[0, 1].item(), .6 * 16, places=5)
        F.cross_entropy(logits, torch.tensor([0])).backward()
        self.assertTrue(torch.isfinite(vector.grad).all())
        self.assertTrue(torch.isfinite(head.weight.grad).all())
        # The angular branch must also remain monotonic near pi, rather than
        # making an opposite-facing target easier after applying the margin.
        reverse = head(torch.tensor([[-.999, .01]]), torch.tensor([0]))[0, 0]
        base = F.normalize(torch.tensor([[-.999, .01]]), dim=1)[0, 0] * 16
        self.assertLess(reverse, base)

    def test_margin_zero_is_normalized_cosine_and_embedding_scale_invariant(self):
        head = ArcFace(3, 4, margin=0, scale=8)
        x = torch.tensor([[.3, .5, .2], [.2, .1, .8]])
        expected = F.linear(F.normalize(x, dim=1), F.normalize(head.weight, dim=1)) * 8
        self.assertTrue(torch.allclose(head(x, torch.tensor([0, 3])), expected, atol=1e-5))
        self.assertTrue(torch.allclose(head(x * 5, torch.tensor([0, 3])), expected, atol=1e-5))

    def test_retrieval_ranks_distinct_identities_and_fails_missing_enrollment(self):
        gallery = [record('A', 'a'), record('A', 'b'), record('B', 'c')]
        query = [record('B', 'q', source='video')]
        report = retrieval_metrics(np.array([[0., 1.]]), np.array([[1., 0.], [.9, .1], [0., 1.]]), query, gallery)
        self.assertEqual(report['recall_at']['1'], 1.)
        self.assertEqual(report['gallery_identities'], 2)
        self.assertEqual(report['records'][0]['expected_identity'], identity_key(query[0]['identity']))
        with self.assertRaisesRegex(ValueError, 'not enrolled'):
            retrieval_metrics(np.array([[0., 1.]]), np.array([[1., 0.], [.9, .1]]), query, gallery[:2])

    def test_augmentation_is_seeded_rgb_full_card_canvas(self):
        rgb = np.zeros((312, 224, 3), dtype=np.uint8)
        rgb[:, :, 0] = 200
        first = augment_card(rgb, random.Random(42))
        second = augment_card(rgb, random.Random(42))
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, rgb.shape)
        self.assertEqual(first.dtype, np.uint8)
        self.assertGreater(first[150, 100, 0], first[150, 100, 2])


if __name__ == '__main__':
    unittest.main()
