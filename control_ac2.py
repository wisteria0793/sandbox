import socket
import time

def send_and_receive_echonet(ip, epc_list):
    """
    指定されたIPのエアコンに対して、EPCリストのプロパティ読み出し(Get)を要求し、
    結果をパースして返却する
    """
    port = 3610
    # 送信元ポート3610をバインドして応答を受け取る
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(3.0)
    
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', 3610))
    except OSError as e:
        print(f"[警告] Port 3610 bind failed (既にポートが占有されている可能性があります): {e}")
        # bindに失敗しても送信は試みる

    # EHD1=10, EHD2=81, TID=0001, SEOJ=05FF01 (コントローラ), DEOJ=013001 (家庭用エアコン), ESV=62 (Get)
    packet = bytearray([
        0x10, 0x81,  # EHD1, EHD2
        0x00, 0x01,  # TID
        0x05, 0xFF, 0x01,  # SEOJ (コントローラ)
        0x01, 0x30, 0x01,  # DEOJ (エアコン)
        0x62,        # ESV (Get)
        len(epc_list) # OPC
    ])
    
    for epc in epc_list:
        packet.append(epc) # EPC
        packet.append(0x00) # PDC (要求は0)

    try:
        print(f"[{ip}] 送信電文: {packet.hex().upper()}")
        sock.sendto(packet, (ip, port))
        
        # 応答受信
        data, addr = sock.recvfrom(1024)
        print(f"[{ip}] 受信電文: {data.hex().upper()}")
        
        if len(data) < 12:
            print("受信データが短すぎます。")
            return {}

        ehd = data[0:2]
        tid = data[2:4]
        seoj = data[4:7]
        deoj = data[7:10]
        esv = data[10]
        opc = data[11]

        results = {}
        
        # Get_Res (0x72) または一部未対応があった場合の SNA (0x5E) をパースする
        if esv in [0x72, 0x5e, 0x7e]:  # Get_Res (正常応答)
            idx = 12
            for _ in range(opc):
                if idx + 2 > len(data):
                    break
                epc_res = data[idx]
                pdc_res = data[idx+1]
                idx += 2
                
                if idx + pdc_res > len(data):
                    break
                edt_res = data[idx : idx + pdc_res]
                idx += pdc_res
                
                # 1バイトの符号付き整数としてデコード
                if pdc_res == 1:
                    # 符号付き整数にデコード (2の補数)
                    val = edt_res[0]
                    if val >= 128:
                        val -= 256
                    results[epc_res] = val
                else:
                    results[epc_res] = edt_res.hex()
                    
            if esv == 0x5e:
                print("[警告] 一部または全てのプロパティが応答不可(SNA)で返されました。")
        else:
            print(f"[警告] 予期しないESV: 0x{esv:02x}")
            
        return results

    except socket.timeout:
        print("[エラー] 応答タイムアウト")
        return {}
    except Exception as e:
        print("[エラー] 通信エラー:", e)
        return {}
    finally:
        sock.close()

if __name__ == "__main__":
    # エアコンのIPアドレスを指定
    AIRCON_IP = "192.168.11.63"  # テストするエアコンのIPに変更してください
    
    print(f"エアコン {AIRCON_IP} から室温(0xBB)と外気温(0xBE)を取得します...")
    
    # 0xBB (室温相当), 0xBE (外気温)
    props = [0xBB, 0xBE]
    res = send_and_receive_echonet(AIRCON_IP, props)
    
    print("\n=== 取得結果 ===")
    if 0xBB in res:
        # 特殊値チェック
        if res[0xBB] in [125, 126, 127, -127, -128]:
            print(f"室温: 測定不能または未サポート (値: {res[0xBB]})")
        else:
            print(f"室温 (吸込み口温度): {res[0xBB]} ℃")
    else:
        print("室温: 取得失敗")
        
    if 0xBE in res:
        # 特殊値チェック
        if res[0xBE] in [125, 126, 127, -127, -128]:
            print(f"外気温: 測定不能または未サポート (値: {res[0xBE]})")
        else:
            print(f"外気温: {res[0xBE]} ℃")
    else:
        print("外気温: 取得失敗")