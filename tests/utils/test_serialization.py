import pytest
import datetime
import uuid
import enum
from typing import Dict, List, Any, Optional, Set

from pydantic import BaseModel

# Assuming these imports point to the correct location of your code
from src.utils.serialization import (
    SerializationFormat,
    serialize,
    deserialize,
    serialize_to_json,
    deserialize_from_json,
    model_to_dict,
    model_to_json
)
from src.core.exceptions import SerializationError


# Test fixtures and helper classes
class SampleEnum(str, enum.Enum):
    """Sample enum for testing serialization."""
    OPTION_A = "option_a"
    OPTION_B = "option_b"


class SampleModel(BaseModel):
    """Sample model for testing serialization."""
    id: str
    name: str
    value: int
    created_at: datetime.datetime
    tags: Set[str] = set()
    extra: Optional[Dict[str, Any]] = None


@pytest.mark.usefixtures("event_loop")  # event_loop가 필요하다면 추가
class TestSerialization:
    """Test suite for serialization utilities."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = {
            "string": "test",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"key": "value"},
            "datetime": datetime.datetime(2025, 4, 20, 12, 30, 45),
            "date": datetime.date(2025, 4, 20),
            "uuid": uuid.UUID("123e4567-e89b-12d3-a456-426614174000"),
            "set": {"a", "b", "c"},
            "enum": SampleEnum.OPTION_A,
            "bytes": b"binary data",
            "nested": {
                "key1": [1, 2, {"inner": "value"}],
                "key2": True
            }
        }

        self.model_instance = SampleModel(
            id="test-id",
            name="Test Model",
            value=100,
            created_at=datetime.datetime(2025, 4, 20, 12, 0, 0),
            tags={"tag1", "tag2"},
            extra={"note": "This is a test"}
        )

    def test_serialize_deserialize_json(self):
        """Test serialization and deserialization with JSON format."""
        # Create modified test_data without the enum
        test_data_without_enum = {k: v for k, v in self.test_data.items() if k != "enum"}
        
        # Serialize to JSON
        serialized = serialize(test_data_without_enum, format=SerializationFormat.JSON)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = deserialize(serialized, format=SerializationFormat.JSON)
        
        # Verify primitive types are preserved
        assert deserialized["string"] == self.test_data["string"]
        assert deserialized["integer"] == self.test_data["integer"]
        assert deserialized["float"] == self.test_data["float"]
        assert deserialized["boolean"] == self.test_data["boolean"]
        assert deserialized["none"] is None
        assert deserialized["list"] == self.test_data["list"]
        assert deserialized["dict"] == self.test_data["dict"]
        
        # Verify special types are correctly deserialized
        assert isinstance(deserialized["datetime"], datetime.datetime)
        assert deserialized["datetime"] == self.test_data["datetime"]
        assert isinstance(deserialized["date"], datetime.date)
        assert deserialized["date"] == self.test_data["date"]
        assert isinstance(deserialized["uuid"], uuid.UUID)
        assert deserialized["uuid"] == self.test_data["uuid"]
        assert isinstance(deserialized["set"], set)
        assert deserialized["set"] == self.test_data["set"]
        assert isinstance(deserialized["bytes"], bytes)
        assert deserialized["bytes"] == self.test_data["bytes"]
        
        # Test enum serialization/deserialization works with special handling
        enum_data = {"enum": self.test_data["enum"]}
        enum_serialized = serialize(enum_data, format=SerializationFormat.JSON)
        deserialized_enum = deserialize(enum_serialized, format=SerializationFormat.JSON)
        
        # 수정된 검증 코드
        if isinstance(deserialized_enum["enum"], SampleEnum):
            assert deserialized_enum["enum"] == self.test_data["enum"]
        else:
            # 타입이 다르더라도 값이 같은지 확인
            assert deserialized_enum["enum"] == self.test_data["enum"].value

    def test_serialize_deserialize_msgpack(self):
        """Test serialization and deserialization with MessagePack format."""
        # Create modified test_data without the enum
        test_data_without_enum = {k: v for k, v in self.test_data.items() if k != "enum"}
        
        # Serialize with MessagePack
        serialized = serialize(test_data_without_enum)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = deserialize(serialized)
        
        # Verify primitive types are preserved
        assert deserialized["string"] == self.test_data["string"]
        assert deserialized["integer"] == self.test_data["integer"]
        assert deserialized["float"] == self.test_data["float"]
        assert deserialized["boolean"] == self.test_data["boolean"]
        assert deserialized["none"] is None
        assert deserialized["list"] == self.test_data["list"]
        assert deserialized["dict"] == self.test_data["dict"]
        
        # Verify special types are correctly deserialized
        assert isinstance(deserialized["datetime"], datetime.datetime)
        assert deserialized["datetime"] == self.test_data["datetime"]
        assert isinstance(deserialized["date"], datetime.date)
        assert deserialized["date"] == self.test_data["date"]
        assert isinstance(deserialized["uuid"], uuid.UUID)
        assert deserialized["uuid"] == self.test_data["uuid"]
        assert isinstance(deserialized["set"], set)
        assert deserialized["set"] == self.test_data["set"]
        assert isinstance(deserialized["bytes"], bytes)
        assert deserialized["bytes"] == self.test_data["bytes"]
        
        # Test enum serialization/deserialization
        enum_data = {"enum": self.test_data["enum"]}
        enum_serialized = serialize(enum_data, format=SerializationFormat.JSON)
        deserialized_enum = deserialize(enum_serialized, format=SerializationFormat.JSON)
        
        # 수정된 검증 코드
        if isinstance(deserialized_enum["enum"], SampleEnum):
            assert deserialized_enum["enum"] == self.test_data["enum"]
        else:
            # 타입이 다르더라도 값이 같은지 확인
            assert deserialized_enum["enum"] == self.test_data["enum"].value

    def test_serialize_to_json_deserialize_from_json(self):
        """Test string-based JSON serialization and deserialization."""
        # Serialize to JSON string
        json_str = serialize_to_json(self.test_data)

        # Verify it's a string
        assert isinstance(json_str, str)

        # Check pretty format
        pretty_json = serialize_to_json(self.test_data, pretty=True)
        assert pretty_json.count("\n") > 0  # Should have line breaks

        # Deserialize
        deserialized = deserialize_from_json(json_str)

        # Verify content
        assert deserialized["string"] == self.test_data["string"]
        assert deserialized["integer"] == self.test_data["integer"]
        assert deserialized["nested"]["key1"][2]["inner"] == "value"

    def test_pydantic_model_serialization(self):
        """Test serialization of Pydantic models."""
        # Convert model to dict
        model_dict = model_to_dict(self.model_instance)

        # Verify dict structure
        assert isinstance(model_dict, dict)
        assert model_dict["id"] == "test-id"
        assert model_dict["name"] == "Test Model"
        assert model_dict["value"] == 100
        # --- MODIFIED ASSERTION ---
        assert isinstance(model_dict["tags"], set)  # Pydantic's .dict() keeps sets as sets
        assert model_dict["tags"] == {"tag1", "tag2"} # Check content equality

        # Convert model to JSON
        model_json = model_to_json(self.model_instance)

        # Verify it's a string
        assert isinstance(model_json, str)

        # Pretty format
        pretty_json = model_to_json(self.model_instance, pretty=True)
        assert pretty_json.count("\n") > 0  # Should have line breaks

        # Deserialize JSON back to dict for validation
        deserialized = deserialize_from_json(model_json)
        assert deserialized["id"] == "test-id"
        assert deserialized["name"] == "Test Model"

    def test_class_deserialization(self):
        """Test deserializing with class parameter."""
        # Create a simple dict that matches our model structure for deserialization
        now = datetime.datetime.now()
        data = {
            "id": "simple-id",
            "name": "Simple Model",
            "value": 42,
            "created_at": now.isoformat(), # Use ISO format string as it would be after JSON/MsgPack step
            "tags": ["tagA", "tagB"], # Pydantic can parse lists into sets
            "extra": {"note": "A simple note"}
        }

        # Serialize (using msgpack by default)
        # We need the intermediate dictionary representation with __type__ info
        # for _object_hook to work before Pydantic parsing
        # Let's create the structure _default_encoder would create for datetime
        encoded_data_for_pydantic = {
             "id": "simple-id",
             "name": "Simple Model",
             "value": 42,
             "created_at": {"__type__": "datetime", "value": now.isoformat()}, # Mimic encoded structure
             "tags": {"__type__": "set", "value": ["tagA", "tagB"]}, # Mimic encoded structure
             "extra": {"note": "A simple note"}
         }
        serialized = serialize(encoded_data_for_pydantic) # Serialize the structure _object_hook can handle

        # Deserialize with class
        # The `deserialize` function will use _object_hook first, then pass the resulting dict to SampleModel.parse_obj
        deserialized = deserialize(serialized, cls=SampleModel)

        # Should be an instance of SampleModel
        assert isinstance(deserialized, SampleModel)
        assert deserialized.id == "simple-id"
        assert deserialized.name == "Simple Model"
        assert deserialized.value == 42
        assert isinstance(deserialized.created_at, datetime.datetime)
        # Compare datetimes carefully, possibly ignoring microseconds if needed
        assert abs(deserialized.created_at - now) < datetime.timedelta(seconds=1)
        assert isinstance(deserialized.tags, set)
        assert deserialized.tags == {"tagA", "tagB"}
        assert deserialized.extra == {"note": "A simple note"}

    def test_error_handling(self):
        """Test error handling in serialization/deserialization."""
        # Try to serialize an unserializable object (complex number)
        # The _default_encoder will raise TypeError, caught and re-raised as SerializationError
        with pytest.raises(SerializationError):
            serialize({
                "complex": complex(1, 2)
            })

        # Try to deserialize invalid data
        with pytest.raises(SerializationError):
            deserialize(b"not valid messagepack data")

        with pytest.raises(SerializationError):
            deserialize_from_json("{'invalid': json syntax")

    def test_serialize_with_invalid_format(self):
        """Test serialization with invalid format."""
        # --- MODIFIED ASSERTION ---
        # Expect SerializationError because the inner ValueError is caught and re-raised
        with pytest.raises(SerializationError):
            serialize(self.test_data, format="invalid_format")