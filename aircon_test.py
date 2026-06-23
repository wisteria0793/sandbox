import asyncio
from pychonet.lib.udpserver import UDPServer
from pychonet import ECHONETAPIClient as api
from pychonet import Factory

async def run_discovery():
    # 1. UDPサーバーの初期化 (ポート3610で受信待機)
    udp = UDPServer()
    loop = asyncio.get_event_loop()
    udp.run("0.0.0.0", 3610, loop=loop)
    
    # 2. APIクライアントの作成
    server = api(server=udp)
    
    # 3. 探索対象の指定
    # ※ 基本的にはマルチキャスト "224.0.23.0" で自動探索します。
    # ※ もしエアコンのIPアドレス（例: "192.168.11.50"）がすでに分かっている場合は、
    #   "224.0.23.0" をそのIPアドレスに直接書き換えることで、より確実に接続テストが行えます。
    TARGET_IP = "192.168.11.63"
    
    print(f"{TARGET_IP} を使って ECHONET Lite デバイスをスキャン中...")
    discovered = await server.discover(TARGET_IP)
    
    # デバイスが見つからなかった場合（Falseが返ってきた場合）の安全なハンドリング
    if not discovered or isinstance(discovered, bool):
        print("\n[エラー] ECHONET Liteデバイスが見つかりませんでした。")
        print("以下を確認してください:")
        print("  1. エアコンがWi-Fiルーターに接続されているか")
        print("  2. エアコンの設定メニューやアプリで『ECHONET Lite連携』『スマートスピーカー連携』『宅内操作』などの設定がONになっているか")
        print("  3. PCがエアコンと同じルーターのWi-Fi（今回は192.168.11.x）に接続されているか")
        return
        
    print("\n--- 発見したデバイス一覧 ---")
    for ip, classes in discovered.items():
        print(f"\n[機器発見] IPアドレス: {ip}")
        for cls in classes:
            # クラス情報の表示 (エアコンは グループ:1, クラス:48)
            print(f" - クラス情報: グループ={cls[0]}, クラス={cls[1]}, インスタンス={cls[2]}")
            
            # エアコン(1, 48)を見つけたら操作する
            if cls[0] == 1 and cls[1] == 48:
                print("   --> 家庭用エアコンを認識しました。操作テストを開始します...")
                
                # Factory関数を使ってエアコンオブジェクトを生成
                aircon = Factory(ip, server, cls[0], cls[1], cls[2])
                
                # 現在の状態を取得
                await aircon.update()
                print(f"   現在の電源状態 (True=ON, False=OFF): {aircon.status}")
                
                # A. 電源ONテスト
                print("   エアコンの電源を【ON】にします...")
                await aircon.set_operational_status(True)
                await asyncio.sleep(2)
                
                # B. モード「冷房」設定テスト
                print("   運転モードを【冷房 (cool)】に設定します...")
                await aircon.set_operation_mode("cool")
                await asyncio.sleep(2)
                
                # C. 設定温度を「26度」に変更テスト
                print("   設定温度を【26℃】に設定します...")
                await aircon.set_target_temperature(26)
                
                print("   操作コマンドの送信完了。")

if __name__ == "__main__":
    asyncio.run(run_discovery())