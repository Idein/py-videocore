from videocore.mailbox import MailBox

with MailBox() as mb:
    print('firmware revision: %x' % mb.get_firmware_revision())
    print('board model: %x' % mb.get_board_model())
    print('board revision: %x' % mb.get_board_revision())
    print('board serial: %016x' % mb.get_board_serial())
