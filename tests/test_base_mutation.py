import unittest
from mapmanagercore.annotations.base_mutation import AnnotationsBaseMut
from mapmanagercore.loader.base import Loader

class TestAnnotationsBaseMut(unittest.TestCase):

    def new(self):
        return AnnotationsBaseMut(Loader())

    def test_undo_redo_simple_spine(self):
        annotations = self.new()
        annotations.updateSpine(("spine_id", 0), {"z": 0})
        self.assertEqual(annotations._points.loc[("spine_id", 0), "z"], 0)

        # Undo the update
        annotations.undo()
        self.assertNotIn(("spine_id", 0), annotations._points.index)

        # Redo the update
        annotations.redo()
        self.assertEqual(annotations._points.loc[("spine_id", 0), "z"], 0)

        # Redo again (should have no effect)
        annotations.redo()
        self.assertEqual(annotations._points.loc[("spine_id", 0), "z"], 0)

        annotations.updateSpine(("spine_id", 0), {"z": 1})
        self.assertEqual(annotations._points.loc[("spine_id", 0), "z"], 1)
        annotations.undo()
        self.assertEqual(annotations._points.loc[("spine_id", 0), "z"], 0)
        annotations.redo()
        self.assertEqual(annotations._points.loc[("spine_id", 0), "z"], 1)
        annotations.undo()
        self.assertEqual(annotations._points.loc[("spine_id", 0), "z"], 0)

        # Undo twice (should have no effect)
        annotations.undo()
        self.assertNotIn(("spine_id", 0), annotations._points.index)
        annotations.undo()
        self.assertNotIn(("spine_id", 0), annotations._points.index)

    def test_undo_redo_replace(self):
        annotations = self.new()
        # Test replaceLog
        annotations.updateSpine(("spine_id", 0), {"z": 2})
        self.assertEqual(len(annotations._log.operations), 1)
        self.assertEqual(annotations._points.loc[("spine_id", 0), "z"], 2)

        annotations.updateSpine(("spine_id", 0), {"z": 3})
        self.assertEqual(len(annotations._log.operations), 2)
        self.assertEqual(annotations._points.loc[("spine_id", 0), "z"], 3)

        annotations.updateSpine(("spine_id", 0), {"z": 4}, replaceLog=True)
        self.assertEqual(len(annotations._log.operations), 2)
        self.assertEqual(annotations._points.loc[("spine_id", 0), "z"], 4)

        annotations.undo()
        self.assertEqual(annotations._points.loc[("spine_id", 0), "z"], 2)

        annotations.undo()
        self.assertNotIn(("spine_id", 0), annotations._points.index)

        annotations.redo()
        self.assertEqual(annotations._points.loc[("spine_id", 0), "z"], 2)

        annotations.redo()
        self.assertEqual(annotations._points.loc[("spine_id", 0), "z"], 4)

    def test_undo_redo_simple_segment(self):
        annotations = self.new()
        annotations.updateSegment(("segment_id", 0), {"radius": 1})
        self.assertEqual(
            annotations._lineSegments.loc[("segment_id", 0), "radius"], 1)

        # Undo the update
        annotations.undo()
        self.assertNotIn(("segment_id", 0), annotations._lineSegments.index)

        # Redo the update
        annotations.redo()
        self.assertEqual(
            annotations._lineSegments.loc[("segment_id", 0), "radius"], 1)

    def test_delete_spine(self):
        annotations = self.new()
        annotations.updateSpine(("spine_id", 0), {"z": 0})
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

    def test_delete_segment(self):
        annotations = self.new()
        annotations.updateSegment(("segment_id", 0), {"radius": 1})
        self.assertIn(("segment_id", 0), annotations._lineSegments.index)

        # Delete the segment
        annotations.deleteSegment(("segment_id", 0))
        self.assertNotIn(("segment_id", 0), annotations._lineSegments.index)

        # Undo the deletion
        annotations.undo()
        self.assertIn(("segment_id", 0), annotations._lineSegments.index)

        # Redo the deletion
        annotations.redo()
        self.assertNotIn(("segment_id", 0), annotations._lineSegments.index)
        
    def test_multi_segment(self):
        annotations = self.new()
        annotations.updateSegment(("segment_id", 0), {"radius": 1})
        annotations.updateSegment(("segment_id2", 0), {"radius": 2})
        self.assertIn(("segment_id", 0), annotations._lineSegments.index)
        self.assertIn(("segment_id2", 0), annotations._lineSegments.index)
        
        self.assertEqual(annotations._lineSegments.loc[("segment_id", 0), "radius"], 1)
        self.assertEqual(annotations._lineSegments.loc[("segment_id2", 0), "radius"], 2)

        annotations.updateSegment([("segment_id", 0), ("segment_id2", 0)], {"radius": 3})
        self.assertEqual(annotations._lineSegments.loc[("segment_id", 0), "radius"], 3)
        self.assertEqual(annotations._lineSegments.loc[("segment_id2", 0), "radius"], 3)
        
        annotations.undo()
        self.assertEqual(annotations._lineSegments.loc[("segment_id", 0), "radius"], 1)
        self.assertEqual(annotations._lineSegments.loc[("segment_id2", 0), "radius"], 2)
        
        annotations.redo()
        self.assertEqual(annotations._lineSegments.loc[("segment_id", 0), "radius"], 3)
        self.assertEqual(annotations._lineSegments.loc[("segment_id2", 0), "radius"], 3)
        
        # Delete the segment
        annotations.deleteSegment([("segment_id", 0), ("segment_id2", 0)])
        self.assertNotIn(("segment_id", 0), annotations._lineSegments.index)
        self.assertNotIn(("segment_id2", 0), annotations._lineSegments.index)

        # Undo the deletion
        annotations.undo()
        self.assertIn(("segment_id", 0), annotations._lineSegments.index)
        self.assertIn(("segment_id2", 0), annotations._lineSegments.index)

        # Redo the deletion
        annotations.redo()
        self.assertNotIn(("segment_id", 0), annotations._lineSegments.index)
        self.assertNotIn(("segment_id2", 0), annotations._lineSegments.index)


if __name__ == '__main__':
    unittest.main()
