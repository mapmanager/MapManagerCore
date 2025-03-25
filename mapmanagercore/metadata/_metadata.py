import dataclasses
from typing import List, Optional

from mapmanagercore.logger import logger

@dataclasses.dataclass
class _metadataBase:
    """Base class for all metadata dataclass.
    """
    _key:Optional[int] = None  # when added to a dict
    
    def setValue(self, key, value) -> bool:
        """Set a field value.
        
        Returns:
            True if field exists, otherwise False
        """
        try:
            setattr(self, key, value)
            return True
        except (AttributeError) as e:
            logger.warning(e)
            return False
        
    def getValue(self, key) -> Optional[object]:
        """Get a field value.
        
        Returns:
            Field value if field exists, otherwise None

        Notes:
            Returning python None can be misleading, make sure "none" of our fields have default value of None ???
        """
        if not isinstance(key, str):
            logger.mmLog(f'attr must be str, got {type(key)}')
            return
        
        try:
            return getattr(self, key)
        except (AttributeError) as e:
            logger.warning(e)

    def asDict(self) -> dict:
        """Get the dataclass as a python dict.
        """
        return dataclasses.asdict(self)
    
@dataclasses.dataclass
class _metadataList(_metadataBase):
    """Base class for all metadata that holds a list.
    
    This currently includes:
        - TimepointMetadata that has a list of ChannelMetadata (channels)
        - MetadataList that has a list of TimepointMetadata
    """

    # _metadataList: list = dataclasses.field(default_factory=list)
    _metadataList: dict = dataclasses.field(default_factory=dict)
    # _metadataList: dict[_metadataBase] = dataclasses.field(default_factory=dict)

    # abb refactor for dict (add)
    def getNewKey(self):
        """Get a unique key for this metadata list.

        This is 1 based.
        Nope, back to zero based to be compatible with gui
        """
        _keys = list(self._metadataList.keys())
        _max = max(_keys)+1 if _keys else 0
        return _max
    
    # abb refactor for dict
    @property
    def listKeys(self) -> List[int]:
        """Get a list of keys from our _metadataList.
        """
        # return list(range(self.numItems))
        return list(self._metadataList.keys())

    @property
    def numItems(self) -> int:
        """Get the number of items in the list.
        """
        return len(self._metadataList)
    
    def getMetadataItem(self, index : int) -> Optional[object]:
        """Get metadata for one item in the list.
        
        Returns:
            item if index exists, otherwise None

        See also:
            __getitem__(int)
        """
        if index in self.listKeys:
            return self._metadataList[index]
        
    def appendMetadataItem(self, metadata : object) -> int:
        """Append a metadata item.
        
        Returns:
            index of new item (1 based).
        """
        _newKey = self.getNewKey()
        self._metadataList[_newKey] = metadata
        metadata._key = _newKey  # keys are immutable, convenience so items know there key
        return _newKey
    
    def deleteMetadataItem(self, index : int) -> Optional[object]:
        """Remove from index.

        Returns:
            The item removed, otherwise None
        """
        if index in self.listKeys:
            item = self._metadataList.pop(index)
            return item
    
    # TODO: implement this
    def _old_insertMetadataItem(self, index : int, metadata : object) -> Optional[bool]:
        """Insert an item at given index.
        
        Returns
            True on success, otherwise None
        """
        if index in self.listKeys:
            self._metadataList.insert(index, metadata)
            return True

    # TODO: implement this
    def swapMetadataItems(self, srcIndex, dstIndex) -> bool:
        """Swap/move an item in the list.
        
        Returns
            True on success, otherwise False
        """
        if srcIndex not in self.listKeys:
            logger.mmlog(f'src {srcIndex} does not exist')
            return False
        if dstIndex not in self.listKeys:
            logger.mmlog(f'dst {dstIndex} does not exist')
            return False
        
        listKeys = self.listKeys
        src =  listKeys.index(srcIndex)
        dst =  listKeys.index(dstIndex)

        _tupleList = list(self._metadataList.items())
    
        # swap
        _tupleList[src], _tupleList[dst] = _tupleList[dst], _tupleList[src]
            
        self._metadataList = dict(_tupleList)
        
        return True
    
    def setItem(self, index, key, value) -> bool:
        """Set one item (key/value) in list.
        """
        if index in self.listKeys:
            return self._metadataList[index].setValue(key, value)
        else:
            return False
        
    def __getitem__(self, index:int):
        """Limit use, use explicit function getMetadataItem(int)
        """
        if index not in self._metadataList.keys():
            logger.error(f'did not find key:{index}')
            return
        return self._metadataList[index]
    
    def __iter__(self):
        """Iterate over the metadata list.
        """
        for index in self.listKeys:
            yield self._metadataList[index]
