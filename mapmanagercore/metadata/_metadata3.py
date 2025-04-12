import dataclasses
from typing import List, Optional

from mapmanagercore.exceptions import MetadataError

from mapmanagercore.logger import logger

@dataclasses.dataclass
class _metadataBase:
    """Base class for all metadata dataclass.
    """
    
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
            logger.mmLog(f'attr must be str, got {key} {type(key)}')
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

    # abb, I want derived classes to have meaninful names like (timepoints, channels)
    # _metadataList: dict = dataclasses.field(default_factory=dict)
    _key = 'UNDEFINDED'

    @property
    def _metadataList(self) -> dict:
        # getter
        return getattr(self, self._key)
    
    @_metadataList.setter
    def _metadataList(self, new_value):
        """Setter method for the _metadataList attribute."""
        setattr(self, self._key, new_value)

    def keyExists(self, key) -> bool:
        """Return True if key exists.
        
        All keys are strings to make save/load from json easier.
        """
        return key in self._metadataList.keys()
    
    def _getNewKey(self) -> int:
        """Get a unique key for this metadata list.

        All keys are 1 based int.
        """
        _intKeys = list(self._metadataList.keys())
        # _intKeys = list(map(int, self._metadataList.keys()))
        # logger.info(f'_keys:{_keys} {type(_keys)}')
        _newKey = max(_intKeys)+1 if _intKeys else 1
        # _maxStr = str(_max)
        return _newKey
    
    # abb refactor for dict
    @property
    def listKeys(self) -> List[int]:
        """Get a list of keys from our _metadataList.
        """
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
        _newKey = self._getNewKey()
        self._metadataList[_newKey] = metadata
        return _newKey
    
    def deleteMetadataItem(self, index : int) -> Optional[object]:
        """Remove from index.

        Returns:
            The item removed, otherwise None
        """
        if index in self.listKeys:
            item = self._metadataList.pop(index)
            return item
    
    def swapMetadataItems(self, srcIndex, dstIndex) -> MetadataError | bool:
        """Swap/move an item in the list.
        
        Returns
            True on success, otherwise False

        Raises:
            MetadataError
        """
        if srcIndex not in self.listKeys:
            _err = f'src {srcIndex} does not exist, expecting one of {self.listKeys}'
            raise MetadataError(_err)
        if dstIndex not in self.listKeys:
            _err = f'dst {dstIndex} does not exist, expecting one of {self.listKeys}'
            raise MetadataError(_err)
        
        listKeys = self.listKeys
        src =  listKeys.index(srcIndex)
        dst =  listKeys.index(dstIndex)

        _tupleList = list(self._metadataList.items())
    
        # swap
        _tupleList[src], _tupleList[dst] = _tupleList[dst], _tupleList[src]
            
        # remake with new order
        self._metadataList = dict(_tupleList)
        
        return True
    
    def setItem(self, index, key, value) -> bool:
        """Set one item (key/value) in list.
        """
        if index in self.listKeys:
            return self._metadataList[index].setValue(key, value)
        else:
            return False
        
    # def __getitem__(self, index:int):
    #     """Limit use, use explicit function getMetadataItem(int)
    #     """
    #     if index not in self._metadataList.keys():
    #         logger.error(f'did not find key:{index}')
    #         return
    #     return self._metadataList[index]
    
    def __iter__(self):
        """Iterate over the metadata list.
        """
        for index in self.listKeys:
            yield self._metadataList[index]
