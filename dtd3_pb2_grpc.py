# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import dtd3_pb2 as dtd3__pb2


class LearnerStub(object):
    """service def for I/O of learner
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ReadData = channel.unary_stream(
                '/dtd3.Learner/ReadData',
                request_serializer=dtd3__pb2.LearnerRequest.SerializeToString,
                response_deserializer=dtd3__pb2.BufferResponse.FromString,
                )
        self.UpdateNetworks = channel.stream_unary(
                '/dtd3.Learner/UpdateNetworks',
                request_serializer=dtd3__pb2.LearnerSend.SerializeToString,
                response_deserializer=dtd3__pb2.BufferStatus.FromString,
                )


class LearnerServicer(object):
    """service def for I/O of learner
    """

    def ReadData(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateNetworks(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_LearnerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ReadData': grpc.unary_stream_rpc_method_handler(
                    servicer.ReadData,
                    request_deserializer=dtd3__pb2.LearnerRequest.FromString,
                    response_serializer=dtd3__pb2.BufferResponse.SerializeToString,
            ),
            'UpdateNetworks': grpc.stream_unary_rpc_method_handler(
                    servicer.UpdateNetworks,
                    request_deserializer=dtd3__pb2.LearnerSend.FromString,
                    response_serializer=dtd3__pb2.BufferStatus.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'dtd3.Learner', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Learner(object):
    """service def for I/O of learner
    """

    @staticmethod
    def ReadData(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/dtd3.Learner/ReadData',
            dtd3__pb2.LearnerRequest.SerializeToString,
            dtd3__pb2.BufferResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpdateNetworks(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/dtd3.Learner/UpdateNetworks',
            dtd3__pb2.LearnerSend.SerializeToString,
            dtd3__pb2.BufferStatus.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)
