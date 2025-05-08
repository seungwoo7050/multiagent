# tests/test_serialization.py
import pytest
import datetime
import uuid
from enum import Enum
import msgspec
import base64

# 테스트 대상 함수 및 Enum 임포트
from src.utils.serialization import (
    serialize, deserialize, serialize_to_json, deserialize_from_json,
    SerializationFormat
)
from src.config.errors import SystemError, ErrorCode

# 테스트용 Enum 정의
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

# 테스트용 데이터 (시간대 명시)
test_data_dict = {
    "string": "hello world",
    "integer": 123,
    "float": 45.67,
    "boolean": True,
    "list": [1, "a", None, True],
    "nested_dict": {"key": "value"},
    "datetime": datetime.datetime.now(datetime.timezone.utc),
    "date": datetime.date.today(),
    "time": datetime.time(12, 30, 59, tzinfo=datetime.timezone.utc),
    "uuid": uuid.uuid4(),
    "set": {1, 2, 3}, # Set은 msgpack/json에서 list로 변환됨
    "enum": Color.GREEN,
    "bytes": b'binary\x00data'
}

# msgspec Struct 정의 (테스트용)
class Point(msgspec.Struct):
    x: int
    y: int

test_data_msgspec_struct = Point(x=10, y=20)


@pytest.mark.parametrize("format", [SerializationFormat.MSGPACK, SerializationFormat.JSON])
def test_basic_serialization_deserialization_roundtrip(format):
    """기본 데이터 타입의 직렬화-역직렬화 왕복 테스트"""
    encoded = serialize(test_data_dict, format=format, pretty=False)
    assert isinstance(encoded, bytes)
    decoded = deserialize(encoded, format=format, cls=dict)
    assert isinstance(decoded, dict)

    # 기본 타입 검증
    assert decoded["string"] == test_data_dict["string"]
    assert decoded["integer"] == test_data_dict["integer"]
    assert abs(decoded["float"] - test_data_dict["float"]) < 1e-9
    assert decoded["boolean"] == test_data_dict["boolean"]
    assert decoded["list"] == test_data_dict["list"]
    assert decoded["nested_dict"] == test_data_dict["nested_dict"]

    # 복합 타입 검증 (Format별 분기)
    if format == SerializationFormat.MSGPACK:
        # datetime 객체 검증
        assert isinstance(decoded["datetime"], (datetime.datetime, str))
        if isinstance(decoded["datetime"], datetime.datetime):
            assert decoded["datetime"].astimezone(datetime.timezone.utc) == test_data_dict["datetime"].astimezone(datetime.timezone.utc)
        else:  # str인 경우
            dt_from_decoded = datetime.datetime.fromisoformat(decoded["datetime"].replace('Z', '+00:00'))
            assert dt_from_decoded.astimezone(datetime.timezone.utc) == test_data_dict["datetime"].astimezone(datetime.timezone.utc)

        # date 검증
        assert isinstance(decoded["date"], (datetime.date, str))
        if isinstance(decoded["date"], str):
            assert datetime.date.fromisoformat(decoded["date"]) == test_data_dict["date"]
        else:
            assert decoded["date"] == test_data_dict["date"]

        # time 검증 - 시간대 정보 무시하고 시/분/초만 비교
        assert isinstance(decoded["time"], (datetime.time, str))
        if isinstance(decoded["time"], str):
            decoded_time = datetime.time.fromisoformat(decoded["time"].replace('Z', '+00:00'))
        else:
            decoded_time = decoded["time"]
        
        original_time = test_data_dict["time"]
        assert decoded_time.hour == original_time.hour
        assert decoded_time.minute == original_time.minute
        assert decoded_time.second == original_time.second

        # uuid 검증
        assert isinstance(decoded["uuid"], (uuid.UUID, str))
        if isinstance(decoded["uuid"], str):
            assert uuid.UUID(decoded["uuid"]) == test_data_dict["uuid"]
        else:
            assert decoded["uuid"] == test_data_dict["uuid"]

    elif format == SerializationFormat.JSON:
        # JSON은 모두 문자열로 변환될 가능성 있음
        # datetime 검증
        assert isinstance(decoded["datetime"], str)
        dt_from_decoded = datetime.datetime.fromisoformat(decoded["datetime"].replace('Z', '+00:00'))
        assert dt_from_decoded.astimezone(datetime.timezone.utc) == test_data_dict["datetime"].astimezone(datetime.timezone.utc)

        # date 검증
        assert isinstance(decoded["date"], str)
        assert datetime.date.fromisoformat(decoded["date"]) == test_data_dict["date"]

        # time 검증 - 시간대 정보 무시하고 시/분/초만 비교
        assert isinstance(decoded["time"], str)
        # 'Z' 또는 '+00:00' 등의 시간대 표시가 있을 수 있으므로 안전하게 변환
        try:
            decoded_time = datetime.time.fromisoformat(decoded["time"].replace('Z', '+00:00'))
        except ValueError:
            # 시간대 정보가 없는 경우
            decoded_time = datetime.time.fromisoformat(decoded["time"])
        
        original_time = test_data_dict["time"]
        assert decoded_time.hour == original_time.hour
        assert decoded_time.minute == original_time.minute
        assert decoded_time.second == original_time.second

        # uuid 검증
        assert isinstance(decoded["uuid"], str)
        assert uuid.UUID(decoded["uuid"]) == test_data_dict["uuid"]

    # set, enum, bytes 검증
    assert isinstance(decoded["set"], list)
    assert set(decoded["set"]) == test_data_dict["set"]
    assert decoded["enum"] == test_data_dict["enum"].value
    assert isinstance(decoded["bytes"], bytes)
    assert decoded["bytes"] == test_data_dict["bytes"]


def test_pickle_serialization_deserialization_roundtrip():
    """Pickle 직렬화-역직렬화 왕복 테스트"""
    complex_data = {"point": test_data_msgspec_struct, "color": Color.BLUE, "dt": test_data_dict["datetime"]}
    encoded = serialize(complex_data, format=SerializationFormat.PICKLE)
    decoded = deserialize(encoded, format=SerializationFormat.PICKLE)
    assert decoded["point"] == complex_data["point"]
    assert decoded["color"] == complex_data["color"]
    assert decoded["dt"] == complex_data["dt"]


def test_msgspec_struct_serialization_deserialization():
    """msgspec Struct 직렬화-역직렬화 테스트"""
    encoded = serialize(test_data_msgspec_struct, format=SerializationFormat.MSGPACK)
    decoded = deserialize(encoded, format=SerializationFormat.MSGPACK, cls=Point)
    assert isinstance(decoded, Point)
    assert decoded == test_data_msgspec_struct


def test_serialize_deserialize_json_string():
    """JSON 문자열 변환 함수 왕복 테스트"""
    json_string = serialize_to_json(test_data_dict, pretty=False)
    assert isinstance(json_string, str)
    decoded = deserialize_from_json(json_string, cls=dict)

    # 검증 로직
    assert decoded["string"] == test_data_dict["string"]
    assert uuid.UUID(decoded["uuid"]) == test_data_dict["uuid"]
    assert set(decoded["set"]) == test_data_dict["set"]
    assert decoded["enum"] == test_data_dict["enum"].value
    assert isinstance(decoded["bytes"], bytes)
    assert decoded["bytes"] == test_data_dict["bytes"]


def test_pretty_json_serialization():
    """Pretty JSON 직렬화 테스트"""
    json_string = serialize_to_json(test_data_dict, pretty=True)
    assert isinstance(json_string, str)
    assert "\n" in json_string  # pretty=True이면 줄바꿈이 있어야 함
    decoded = deserialize_from_json(json_string, cls=dict)
    assert decoded["string"] == test_data_dict["string"]
    assert isinstance(decoded["bytes"], bytes)
    assert decoded["bytes"] == test_data_dict["bytes"]


def test_empty_data_deserialization():
    """빈 데이터 역직렬화 예외 테스트"""
    with pytest.raises(SystemError):
        deserialize(b"", format=SerializationFormat.JSON)
    with pytest.raises(SystemError):
        deserialize(b"", format=SerializationFormat.MSGPACK)
    with pytest.raises(SystemError):
        deserialize(b"", format=SerializationFormat.PICKLE)