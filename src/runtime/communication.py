"""
网络通信模块
实现边缘设备和云端服务器之间的Socket通信
"""

import socket
import struct
import time


class DeviceClient:
    """边缘设备端通信客户端"""
    
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = None
        self.bandwidth_history = []
    
    def connect(self):
        """连接到云端服务器"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.server_ip, self.server_port))
    
    def send_data(self, data, alpha, split_point):
        """
        发送压缩数据和配置
        
        Args:
            data: bytes, 压缩后的数据
            alpha: float, 剪枝率
            split_point: int, 分割点
        
        Returns:
            float: 带宽(bps)
        """
        start_time = time.time()
        
        # 发送配置 (alpha: float, split_point: float)
        config = struct.pack('ff', alpha, float(split_point))
        self.socket.sendall(config)
        
        # 发送数据长度
        data_len = len(data)
        self.socket.sendall(struct.pack('I', data_len))
        
        # 发送数据
        self.socket.sendall(data)
        
        # 计算带宽
        elapsed = time.time() - start_time
        if elapsed > 0:
            bandwidth = (data_len * 8) / elapsed  # bps
            self.bandwidth_history.append(bandwidth)
        else:
            bandwidth = 0
        
        return bandwidth
    
    def receive_result(self):
        """
        接收推理结果
        
        Returns:
            bytes: 结果数据
        """
        # 接收结果长度
        result_len = struct.unpack('I', self._recv_exactly(4))[0]
        # 接收结果数据
        result_data = self._recv_exactly(result_len)
        return result_data
    
    def _recv_exactly(self, n):
        """精确接收n字节"""
        data = b''
        while len(data) < n:
            packet = self.socket.recv(n - len(data))
            if not packet:
                raise ConnectionError("Connection closed")
            data += packet
        return data
    
    def close(self):
        """关闭连接"""
        if self.socket:
            self.socket.close()


class CloudServer:
    """云端服务器通信服务端"""
    
    def __init__(self, port):
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('0.0.0.0', port))
        self.socket.listen(5)
    
    def accept_connection(self):
        """
        接受客户端连接
        
        Returns:
            tuple: (conn, addr)
        """
        conn, addr = self.socket.accept()
        return conn, addr
    
    def receive_data(self, conn):
        """
        接收数据和配置
        
        Args:
            conn: socket连接
        
        Returns:
            tuple: (data, alpha, split_point)
        """
        # 接收配置
        config_data = self._recv_exactly(conn, 8)
        alpha, split_point = struct.unpack('ff', config_data)
        
        # 接收数据长度
        data_len = struct.unpack('I', self._recv_exactly(conn, 4))[0]
        
        # 接收数据
        data = self._recv_exactly(conn, data_len)
        
        return data, alpha, int(split_point)
    
    def send_result(self, conn, result_data):
        """
        发送结果
        
        Args:
            conn: socket连接
            result_data: bytes, 结果数据
        """
        # 发送结果长度
        conn.sendall(struct.pack('I', len(result_data)))
        # 发送结果
        conn.sendall(result_data)
    
    def _recv_exactly(self, conn, n):
        """精确接收n字节"""
        data = b''
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:
                raise ConnectionError("Connection closed")
            data += packet
        return data
    
    def close(self):
        """关闭服务器"""
        self.socket.close()


if __name__ == "__main__":
    import threading
    
    print("Testing communication module...")
    
    # 服务端线程
    def server_thread():
        server = CloudServer(9999)
        print("Server: Listening on port 9999...")
        conn, addr = server.accept_connection()
        print(f"Server: Connection from {addr}")
        
        data, alpha, split = server.receive_data(conn)
        print(f"Server: Received {len(data)} bytes, α={alpha:.4f}, split={split}")
        
        server.send_result(conn, b"test_result_from_server")
        print("Server: Result sent")
        
        conn.close()
        server.close()
    
    # 启动服务端
    t = threading.Thread(target=server_thread)
    t.start()
    
    time.sleep(0.5)
    
    # 客户端
    print("\nClient: Connecting to server...")
    client = DeviceClient('127.0.0.1', 9999)
    client.connect()
    print("Client: Connected")
    
    test_data = b"test_data" * 1000
    bw = client.send_data(test_data, 0.25, 12)
    print(f"Client: Data sent, bandwidth: {bw / 1e6:.2f} Mbps")
    
    result = client.receive_result()
    print(f"Client: Received result: {result}")
    
    client.close()
    print("Client: Connection closed")
    
    t.join()
    print("\nCommunication module tests completed!")
