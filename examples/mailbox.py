from pyvc.mailbox import MailBox

with MailBox() as mb:
    firmware_rev   = mb.get_firmware_revision()
    board_model    = mb.get_board_model()
    board_revision = mb.get_board_revision()
    board_serial   = mb.get_board_serial()
    print('firmware revision: %x' % firmware_rev)
    print('board model: %x' % board_model)
    print('board revision: %x' % board_revision)
    print('board serial: %016x' % board_serial)
