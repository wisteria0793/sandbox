import socket
import time

class SimpleEchonetAircon:
    def __init__(self, ip):
        self.ip = ip
        self.port = 3610

    def _send_only(self, packet):
        """パケットを送信するだけで応答を待たない (操作・設定用)"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(packet, (self.ip, self.port))
            return True
        except Exception as e:
            print("送信エラー:", e)
            return False
        finally:
            sock.close()

    def _send_and_receive(self, packet):
        """送信元ポート3610をバインドして送信し、応答を受信する (状態取得用)"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2.0) # 2秒間応答を待つ
        try:
            # 送信元のポートも 3610 にバインドする (エアコンからの応答を受け取るために必須)
            try:
                sock.bind(('', 3610))
            except OSError:
                # すでにポート3610が別のプロセスで使用されている場合はそのままOS自動割り当てで行う
                pass
                
            sock.sendto(packet, (self.ip, self.port))
            data, addr = sock.recvfrom(1024)
            return data
        except socket.timeout:
            return None
        except Exception as e:
            print("通信エラー:", e)
            return None
        finally:
            sock.close()

    def set_power(self, power_on):
        """電源状態を設定する (True: ON, False: OFF)"""
        val = b'\x30' if power_on else b'\x31'
        packet = b'\x10\x81\x00\x01\x05\xFF\x01\x01\x30\x01\x61\x01\x80\x01' + val
        return self._send_only(packet)

    def set_temperature(self, temp):
        """設定温度を変更する (16〜30度)"""
        if not (16 <= temp <= 30):
            raise ValueError("温度は16度から30度の間で指定してください。")
        val = bytes([int(temp)])
        packet = b'\x10\x81\x00\x02\x05\xFF\x01\x01\x30\x01\x61\x01\xB3\x01' + val
        return self._send_only(packet)

    def set_mode(self, mode):
        """運転モードを変更する ('cool', 'heat', 'dry', 'fan', 'auto')"""
        modes = {
            'auto': b'\x41',
            'cool': b'\x42',
            'heat': b'\x43',
            'dry': b'\x44',
            'fan': b'\x45'
        }
        if mode not in modes:
            raise ValueError(f"不明なモードです: {mode}")
        packet = b'\x10\x81\x00\x03\x05\xFF\x01\x01\x30\x01\x61\x01\xB0\x01' + modes[mode]
        return self._send_only(packet)

    def get_status(self):
        """現在のエアコンの状態（電源、モード、設定温度）を直接取得する"""
        pkt_power = b'\x10\x81\x00\x06\x05\xFF\x01\x01\x30\x01\x62\x01\x80\x00' # 電源 (EPC: 0x80)
        pkt_mode  = b'\x10\x81\x00\x07\x05\xFF\x01\x01\x30\x01\x62\x01\xB0\x00' # モード (EPC: 0xB0)
        pkt_temp  = b'\x10\x81\x00\x08\x05\xFF\x01\x01\x30\x01\x62\x01\xB3\x00' # 温度 (EPC: 0xB3)
        
        status = {}
        
        # 1. 電源状態の読み取り
        res = self._send_and_receive(pkt_power)
        if res and len(res) >= 15:
            status['power'] = "ON" if res[14] == 0x30 else "OFF"
            
        # 2. 運転モードの読み取り
        res = self._send_and_receive(pkt_mode)
        if res and len(res) >= 15:
            modes_map = {0x41: 'auto', 0x42: 'cool', 0x43: 'heat', 0x44: 'dry', 0x45: 'fan'}
            status['mode'] = modes_map.get(res[14], 'unknown')
            
        # 3. 設定温度の読み取り
        res = self._send_and_receive(pkt_temp)
        if res and len(res) >= 15:
            status['temperature'] = int(res[14])
            
        return status

# --- テスト実行 ---
if __name__ == "__main__":
    ac = SimpleEchonetAircon("192.168.11.63")
    
    # 1. 現在の状態の取得
    print("現在の状態を取得中...")
    status = ac.get_status()
    print("現在のエアコン状態:", status)
    
    time.sleep(2)
    
    # ★最初まず電源をONにします（ここでエアコンがピッとなって起動するはずです）
    # print("\nエアコンの電源を【ON】にします...")
    ac.set_power(True)
    
    time.sleep(3) # エアコンが起動するのを少し待つ
    
    # 2. 温度設定を 27度 に変更
    # print("\n設定温度を【27℃】に変更します...")
    # if ac.set_temperature(22):
    #     print("温度設定コマンドを送信しました。")
    # else:
    #     print("送信失敗")
        
    # time.sleep(2)
    
    # # 3. 変更後のステータスを再取得
    # print("\n最新の状態を取得中...")
    # new_status = ac.get_status()
    # print("変更後のエアコン状態:", new_status)