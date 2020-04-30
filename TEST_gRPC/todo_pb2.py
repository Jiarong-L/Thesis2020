# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: todo.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='todo.proto',
  package='todoPackage',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\ntodo.proto\x12\x0btodoPackage\"D\n\x0bmodelWeight\x12\r\n\x05model\x18\x01 \x01(\t\x12\x12\n\nclientname\x18\x02 \x01(\t\x12\x12\n\nclientsize\x18\x03 \x01(\x05\"\x1b\n\nmyResponse\x12\r\n\x05value\x18\x01 \x01(\x05\"\x06\n\x04void2\xbd\x01\n\x05\x46\x65\x64ML\x12:\n\tgetWeight\x12\x11.todoPackage.void\x1a\x18.todoPackage.modelWeight0\x01\x12\x41\n\nsendWeight\x12\x18.todoPackage.modelWeight\x1a\x17.todoPackage.myResponse(\x01\x12\x35\n\x07reCheck\x12\x11.todoPackage.void\x1a\x17.todoPackage.myResponseb\x06proto3'
)




_MODELWEIGHT = _descriptor.Descriptor(
  name='modelWeight',
  full_name='todoPackage.modelWeight',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model', full_name='todoPackage.modelWeight.model', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='clientname', full_name='todoPackage.modelWeight.clientname', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='clientsize', full_name='todoPackage.modelWeight.clientsize', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=27,
  serialized_end=95,
)


_MYRESPONSE = _descriptor.Descriptor(
  name='myResponse',
  full_name='todoPackage.myResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='todoPackage.myResponse.value', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=97,
  serialized_end=124,
)


_VOID = _descriptor.Descriptor(
  name='void',
  full_name='todoPackage.void',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=126,
  serialized_end=132,
)

DESCRIPTOR.message_types_by_name['modelWeight'] = _MODELWEIGHT
DESCRIPTOR.message_types_by_name['myResponse'] = _MYRESPONSE
DESCRIPTOR.message_types_by_name['void'] = _VOID
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

modelWeight = _reflection.GeneratedProtocolMessageType('modelWeight', (_message.Message,), {
  'DESCRIPTOR' : _MODELWEIGHT,
  '__module__' : 'todo_pb2'
  # @@protoc_insertion_point(class_scope:todoPackage.modelWeight)
  })
_sym_db.RegisterMessage(modelWeight)

myResponse = _reflection.GeneratedProtocolMessageType('myResponse', (_message.Message,), {
  'DESCRIPTOR' : _MYRESPONSE,
  '__module__' : 'todo_pb2'
  # @@protoc_insertion_point(class_scope:todoPackage.myResponse)
  })
_sym_db.RegisterMessage(myResponse)

void = _reflection.GeneratedProtocolMessageType('void', (_message.Message,), {
  'DESCRIPTOR' : _VOID,
  '__module__' : 'todo_pb2'
  # @@protoc_insertion_point(class_scope:todoPackage.void)
  })
_sym_db.RegisterMessage(void)



_FEDML = _descriptor.ServiceDescriptor(
  name='FedML',
  full_name='todoPackage.FedML',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=135,
  serialized_end=324,
  methods=[
  _descriptor.MethodDescriptor(
    name='getWeight',
    full_name='todoPackage.FedML.getWeight',
    index=0,
    containing_service=None,
    input_type=_VOID,
    output_type=_MODELWEIGHT,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='sendWeight',
    full_name='todoPackage.FedML.sendWeight',
    index=1,
    containing_service=None,
    input_type=_MODELWEIGHT,
    output_type=_MYRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='reCheck',
    full_name='todoPackage.FedML.reCheck',
    index=2,
    containing_service=None,
    input_type=_VOID,
    output_type=_MYRESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_FEDML)

DESCRIPTOR.services_by_name['FedML'] = _FEDML

# @@protoc_insertion_point(module_scope)
