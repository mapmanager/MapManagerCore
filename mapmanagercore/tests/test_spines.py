import unittest
from mapmanagercore.annotations.mutation import AnnotationsBaseMut
from mapmanagercore.lazy_geo_pd_images.loader.base import ImageLoader
from mapmanagercore.schemas.spine import Spine


class TestAnnotationsBaseMut(unittest.TestCase):

    def new(self):
        return AnnotationsBaseMut(ImageLoader())

    def test_undo_redo_simple_spine(self):
        annotations = self.new()
        annotations.updateSpine(("spine_id", 0), Spine(z=0))
        self.assertEqual(annotations.points[("spine_id", 0), "z"], 0)

        # Undo the update
        annotations.undo()
        self.assertNotIn(("spine_id", 0), annotations.points.index)

        # Redo the update
        annotations.redo()
        self.assertEqual(annotations.points[("spine_id", 0), "z"], 0)

        # Redo again (should have no effect)
        annotations.redo()
        self.assertEqual(annotations.points[("spine_id", 0), "z"], 0)

        annotations.updateSpine(("spine_id", 0), Spine(z=1))
        self.assertEqual(annotations.points[("spine_id", 0), "z"], 1)
        annotations.undo()
        self.assertEqual(annotations.points[("spine_id", 0), "z"], 0)
        annotations.redo()
        self.assertEqual(annotations.points[("spine_id", 0), "z"], 1)
        annotations.undo()
        self.assertEqual(annotations.points[("spine_id", 0), "z"], 0)

        # Undo twice (should have no effect)
        annotations.undo()
        self.assertNotIn(("spine_id", 0), annotations.points.index)
        annotations.undo()
        self.assertNotIn(("spine_id", 0), annotations.points.index)

    def test_undo_redo_replace(self):
        annotations = self.new()
        # Test replaceLog
        annotations.updateSpine(("spine_id", 0), Spine(z=2))
        self.assertEqual(len(annotations._log.operations), 1)
        self.assertEqual(annotations.points[("spine_id", 0), "z"], 2)

        annotations.updateSpine(("spine_id", 0), Spine(z=3))
        self.assertEqual(len(annotations._log.operations), 2)
        self.assertEqual(annotations.points[("spine_id", 0), "z"], 3)

        annotations.updateSpine(("spine_id", 0), Spine(z=4), replaceLog=True)
        self.assertEqual(len(annotations._log.operations), 2)
        self.assertEqual(annotations.points[("spine_id", 0), "z"], 4)

        annotations.undo()
        self.assertEqual(annotations.points[("spine_id", 0), "z"], 2)

        annotations.undo()
        self.assertNotIn(("spine_id", 0), annotations.points.index)

        annotations.redo()
        self.assertEqual(annotations.points[("spine_id", 0), "z"], 2)

        annotations.redo()
        self.assertEqual(annotations.points[("spine_id", 0), "z"], 4)

    def test_delete_spine(self):
        annotations = self.new()
        annotations.updateSpine(("spine_id", 0), Spine(z=0))
        self.assertIn(("spine_id", 0), annotations._points.index)

        # Delete the spine
        annotations.deleteSpine(("spine_id", 0))
        self.assertNotIn(("spine_id", 0), annotations._points.index)

        # Undo the deletion
        annotations.undo()
        self.assertIn(("spine_id", 0), annotations._points.index)

        # Redo the deletion
        annotations.redo()
        self.assertNotIn(("spine_id", 0), annotations._points.index)


if __name__ == '__main__':
    unittest.main()
