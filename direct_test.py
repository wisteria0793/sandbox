import socket

# 自動探索で見つかったエオリアエアコンのIP
AIRCON_IP = "192.168.11.63"
PORT = 3610

# ECHONET Lite 電源ONコマンドの生データ
CMD_POWER_ON = b'\x10\x81\x00\x01\x05\xFF\x01\x01\x30\x01\x61\x01\x80\x01\x30'

print(f"Sending direct ECHONET Lite Power ON packet to {AIRCON_IP}...")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock.sendto(CMD_POWER_ON, (AIRCON_IP, PORT))
    print("送信完了。エアコンから『ピッ』という電子音が鳴りましたか？")
finally:
    sock.close()