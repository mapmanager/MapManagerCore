import dataclasses
from typing import List, Optional, Any

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
            logger.error(f'AttributeError:{e}')
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
            logger.error(e)

    def reset_to_defaults(self):
        """Reset all fields to default values.
        """
        for field_info in dataclasses.fields(self):
            # logger.info(f'name:{field_info.name} default:{field_info.default}')
            setattr(self, field_info.name, field_info.default)

    def __getitem__(self, key: str) -> Any:
        def _fieldNames(self) -> List[str]:
            names = []
            for oneField in dataclasses.fields(self):
                names.append(oneField.name)
            return names
        
        try:
            return getattr(self, key)
        except (AttributeError) as e:
            logger.error(e)
            logger.error(f'available keys are: {_fieldNames()}')

    def asDict(self) -> dict:
        """Get the dataclass as a python dict.

        Returns key:value pairs
        """
        return dataclasses.asdict(self)
            
    def updateDromDict(self, data: dict[str, Any]):
        """Update fields from dict.

        Dict must be just key:value pairs.
        """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f'did not find key:{key}')

    def to_dict_with_metadata(self) -> dict:
        """Get a dict of all fields.
        
        Returns
        -------
        Dict with field name keys, each key is a dict of keys:
            currentValue
            defaultValue
            description
            title
            type
        """
        retDict = {}
        for oneField in dataclasses.fields(self):
            # print(f'oneField:{oneField}')
            
            name = oneField.name
            typeStr = oneField.type.__name__  # like (int, float, bool)
 
            value = self.getValue(name)
            default = oneField.default
            
            # logger.warning(f'name:{name} typeStr:{typeStr} value:{value} default:{default}')

            # all metadata must contain 'description' key
            try:
                metadata = dict(oneField.metadata)
                # pprint(metadata)
            except (KeyError) as e:
                metadata = {}
            
            description = metadata['description'] if 'description' in metadata.keys() else ''
            title = metadata['title'] if 'title' in metadata.keys() else ''
            
            retDict[name] = {
                'currentValue': value,
                'defaultValue': default,
                'description': description,
                'title': title,
                'type': typeStr,
            }
            # print(name, value, _type, default, metadata)
        
        return retDict

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
    
    # generated by cursor ai 202508
    def moveItem(self, srcKey: int, dstKey: int) -> MetadataError | bool:
        """Move an item from one position to another.
        
        Moves the item at srcKey to the position before dstKey.
        If dstKey is None, the item is moved to the end.
        Keys remain persistent - only the order changes.
        
        Args:
            srcKey: The key of the item to move
            dstKey: The key to insert before, or None to move to the end
            
        Returns:
            True on success, otherwise False
            
        Raises:
            MetadataError: If srcKey doesn't exist or dstKey is invalid
        """
        if srcKey not in self.listKeys:
            _err = f'src {srcKey} does not exist, expecting one of {self.listKeys}'
            raise MetadataError(_err)
        
        # If dstKey is None, move to the end
        if dstKey is None:
            # Get the item to move
            item_to_move = self._metadataList[srcKey]
            
            # Remove from current position
            del self._metadataList[srcKey]
            
            # Add to the end with the same key
            self._metadataList[srcKey] = item_to_move
            
            return True
        
        # Validate dstKey exists
        if dstKey not in self.listKeys:
            _err = f'dst {dstKey} does not exist, expecting one of {self.listKeys}'
            raise MetadataError(_err)
        
        # Don't do anything if moving to the same position
        if srcKey == dstKey:
            return True
        
        # Get current order of keys
        listKeys = self.listKeys
        src_index = listKeys.index(srcKey)
        dst_index = listKeys.index(dstKey)
        
        # Convert to list of tuples for manipulation
        _tupleList = list(self._metadataList.items())
        
        # Remove the item to move
        item_to_move = _tupleList.pop(src_index)
        
        # Calculate new destination index after removal
        if src_index < dst_index:
            # Item was removed before destination, so destination index is now one less
            new_dst_index = dst_index - 1
        else:
            # Item was removed after destination, so destination index stays the same
            new_dst_index = dst_index
        
        # Insert the item at the new destination with the same key
        _tupleList.insert(new_dst_index, item_to_move)
        
        # Rebuild the dictionary with new order
        self._metadataList = dict(_tupleList)
        
        return True

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

        # logger.info(f"self._metadataList {self._metadataList}")
        _tupleList = list(self._metadataList.items())

        # swap
        # logger.info(f"source index {_tupleList[src][0]}, destination dict{_tupleList[dst][1]}")
        # logger.info(f"destination index {_tupleList[dst][0]}, source dict {_tupleList[src][1]}")
        # _tupleList[src], _tupleList[dst] = _tupleList[dst], _tupleList[src]
        
        # abj: maintain src index and dest index
        _tupleList[src], _tupleList[dst] = (_tupleList[src][0], _tupleList[dst][1]), (_tupleList[dst][0], _tupleList[src][1])

        # logger.info(f"revised tuplelist {_tupleList}")
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
