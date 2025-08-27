import pytest
import numpy as np

from mapmanagercore.schemas import Spine, Segment


class TestSchemaColumnMethods:
    """Test the column discovery methods in the Schema class."""
    
    def test_spine_get_column_names_basic_functionality(self):
        """Test that Spine.getColumnNames() returns a list of strings."""
        all_columns = Spine.getColumnNames()
        
        # Should return a list
        assert isinstance(all_columns, list)
        assert len(all_columns) > 0
        
        # Should contain only strings
        assert all(isinstance(col, str) for col in all_columns)
    
    def test_segment_get_column_names_basic_functionality(self):
        """Test that Segment.getColumnNames() returns a list of strings."""
        all_columns = Segment.getColumnNames()
        
        # Should return a list
        assert isinstance(all_columns, list)
        assert len(all_columns) > 0
        
        # Should contain only strings
        assert all(isinstance(col, str) for col in all_columns)
    
    def test_spine_get_column_names_parameter_combinations(self):
        """Test Spine.getColumnNames() with different parameter combinations."""
        # Get all columns
        all_columns = Spine.getColumnNames()
        assert isinstance(all_columns, list)
        assert len(all_columns) > 0
        
        # Get only computed columns
        computed_columns = Spine.getColumnNames(include_computed=True, include_basic=False)
        assert isinstance(computed_columns, list)
        
        # Get only basic columns
        basic_columns = Spine.getColumnNames(include_computed=False, include_basic=True)
        assert isinstance(basic_columns, list)
        
        # Get no columns (edge case)
        no_columns = Spine.getColumnNames(include_computed=False, include_basic=False)
        assert no_columns == []
    
    def test_segment_get_column_names_parameter_combinations(self):
        """Test Segment.getColumnNames() with different parameter combinations."""
        # Get all columns
        all_columns = Segment.getColumnNames()
        assert isinstance(all_columns, list)
        assert len(all_columns) > 0
        
        # Get only computed columns
        computed_columns = Segment.getColumnNames(include_computed=True, include_basic=False)
        assert isinstance(computed_columns, list)
        
        # Get only basic columns
        basic_columns = Segment.getColumnNames(include_computed=False, include_basic=True)
        assert isinstance(basic_columns, list)
        
        # Get no columns (edge case)
        no_columns = Segment.getColumnNames(include_computed=False, include_basic=False)
        assert no_columns == []
    

    
    def test_spine_default_parameters(self):
        """Test that Spine.getColumnNames() works with default parameters."""
        # Should work with no parameters (defaults to include_computed=True, include_basic=True)
        all_columns = Spine.getColumnNames()
        assert isinstance(all_columns, list)
        assert len(all_columns) > 0
        
        # Should be the same as explicitly setting both to True
        explicit_all = Spine.getColumnNames(include_computed=True, include_basic=True)
        assert all_columns == explicit_all
    
    def test_segment_default_parameters(self):
        """Test that Segment.getColumnNames() works with default parameters."""
        # Should work with no parameters (defaults to include_computed=True, include_basic=True)
        all_columns = Segment.getColumnNames()
        assert isinstance(all_columns, list)
        assert len(all_columns) > 0
        
        # Should be the same as explicitly setting both to True
        explicit_all = Segment.getColumnNames(include_computed=True, include_basic=True)
        assert all_columns == explicit_all
    
    def test_spine_schema_independence(self):
        """Test that Spine schema methods work independently of any data."""
        # These should work even without any instances or data
        columns1 = Spine.getColumnNames()
        columns2 = Spine.getColumnNames()
        
        # Should be consistent
        assert columns1 == columns2
        
        # Should be the same across multiple calls
        for _ in range(3):
            assert Spine.getColumnNames() == columns1
    
    def test_segment_schema_independence(self):
        """Test that Segment schema methods work independently of any data."""
        # These should work even without any instances or data
        columns1 = Segment.getColumnNames()
        columns2 = Segment.getColumnNames()
        
        # Should be consistent
        assert columns1 == columns2
        
        # Should be the same across multiple calls
        for _ in range(3):
            assert Segment.getColumnNames() == columns1


if __name__ == "__main__":
    pytest.main([__file__])
