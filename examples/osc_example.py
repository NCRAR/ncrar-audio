import time

from ncrar_audio.babyface import Babyface


def main():
    device = Babyface()
    device.set_volume(-6)
    time.sleep(0.1)
    print(device._volume_db)

    #messages = []
    #for i in range(12):
    #    messages.append(['/setSubmix', i])
    #    messages.append(['/1/busPla', i])

    submix = 1
    print(f'Set submix {submix}')
    device.osc_client.send_messages([['/setSubmix', submix]])
    for channel in range(12):
        device.osc_client.send_messages([['/1/busOutput', 1]])
        device.osc_client.send_messages([[f'/1/volume{channel+1}', 0]])
        device.osc_client.send_messages([['/1/busPlayback', 1]])
        device.osc_client.send_messages([[f'/1/volume{channel+1}', 0]])
        time.sleep(0.5)

    return


    messages = []
    for submix in range(12):
        messages.extend([
            ['/busOutput', 1.0],
            [f'/1/volume{submix+1}', 0.818],
            ["/1/busPlayback", 1.0],
        ])
        for channel in range(12):
            volume = 0.818 if channel == submix else 0
            messages.append([f"/1/volume{channel+1}", volume])

    print(messages)

    #set_submix = [
    #        ["/setSubmix", 0.0],
    #        ["/1/volume1",0.818],
    #        ["/1/busOutput", 1.0],
    #        ["/1/volume1",0.818],
    #        #["/setsubmix", 2.0],
    #        #["/1/busOutput", 1.0],
    #        #["/1/volume3",0.818],
    #]
    #device.osc_client.send_messages(messages)
    #device.send_messages([
    #    ['/1/setSubmix', 0],
    #    ['1/busPlayback', 1],
    #    ['1/busPlayback', 1],



if __name__ == '__main__':
    main()
