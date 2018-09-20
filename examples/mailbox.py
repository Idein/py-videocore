from videocore.mailbox import MailBox

with MailBox() as mb:
    print('firmware revision: {:x}'.format(mb.get_firmware_revision()))
    print('board model:       {:x}'.format(mb.get_board_model()))
    print('board revision:    {:x}'.format(mb.get_board_revision()))
    print('board serial:      {:016x}'.format(mb.get_board_serial()))
    print('temperature:       {:.2f}'.format(float(mb.get_temperature(0)[1])/1000.0))
    print('max temperature:   {:.2f}'.format(float(mb.get_max_temperature(0)[1])/1000.0))
    print('throttled:         {:#x}'.format( int(mb.get_throttled() )))
