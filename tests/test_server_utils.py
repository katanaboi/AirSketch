import pytest
import socket
import struct
from unittest.mock import patch, MagicMock
from AirSketch_server import recv_all, send_message, _lan_ip

def test_recv_all_success():
    mock_sock = MagicMock()
    mock_sock.recv.side_effect = [b'hello', b' world']
    
    result = recv_all(mock_sock, 11)
    assert result == b'hello world'
    assert mock_sock.recv.call_count == 2

def test_recv_all_disconnect():
    mock_sock = MagicMock()
    mock_sock.recv.return_value = b'' # Connection closed
    
    result = recv_all(mock_sock, 10)
    assert result is None

def test_send_message_success():
    mock_sock = MagicMock()
    message = "test message"
    message_bytes = message.encode('utf-8')
    length_prefix = struct.pack('<I', len(message_bytes))
    
    send_message(mock_sock, message)
    
    # Should call sendall twice: once for length, once for content
    mock_sock.sendall.assert_any_call(length_prefix)
    mock_sock.sendall.assert_any_call(message_bytes)

def test_send_message_failure():
    mock_sock = MagicMock()
    mock_sock.sendall.side_effect = ConnectionResetError("Disconnected")
    
    with pytest.raises(ConnectionResetError):
        send_message(mock_sock, "hello")

@patch('socket.socket')
def test_lan_ip(mock_socket_class):
    mock_sock = MagicMock()
    mock_socket_class.return_value = mock_sock
    mock_sock.getsockname.return_value = ("192.168.1.100", 12345)
    
    ip = _lan_ip()
    
    assert ip == "192.168.1.100"
    assert mock_sock.connect.called
    # Connects to 8.8.8.8 to determine local IP
    assert mock_sock.connect.call_args[0][0] == ("8.8.8.8", 80)
