"""Wrapper of mailBox property interface."""

import os
from array import array
from struct import calcsize, pack_into, unpack_from
from fcntl import ioctl

IOCTL_MAILBOX = 0xC0046400   # _IOWR(100, 0, char *)

# Use static constant length buffer to avoid unnecessary copy.
# See the document of fcntl.ioctl
IOCTL_BUFSIZE = 1024

# request/response code
PROCESS_REQUEST = 0x00000000
REQUEST_SUCCESS = 0x80000000
PARSE_ERROR     = 0x80000001

class MailBoxException(Exception):
    'Exception related to mailbox property interface.'

class MailBox(object):
    """MailBox Property Interface.

    Implementation of mailbox property interface for VideoCore IV on Raspberry Pi.
    See https://github.com/raspberrypi/firmware/wiki/MailBox-property-interface for details.

    """
    # Device IDs
    DEVICE_SDCARD  = 0x00000000
    DEVICE_UART0   = 0x00000001
    DEVICE_UART1   = 0x00000002
    DEVICE_USB_HCD = 0x00000003
    DEVICE_I2C0    = 0x00000004
    DEVICE_I2C1    = 0x00000005
    DEVICE_I2C2    = 0x00000006
    DEVICE_SPI     = 0x00000007
    DEVICE_CCP2TX  = 0x00000008
    
    # Clock IDs 
    CLOCK_EMMC  = 0x000000001
    CLOCK_UART  = 0x000000002
    CLOCK_ARM   = 0x000000003
    CLOCK_CORE  = 0x000000004
    CLOCK_V3D   = 0x000000005
    CLOCK_H264  = 0x000000006
    CLOCK_ISP   = 0x000000007
    CLOCK_SDRAM = 0x000000008
    CLOCK_PIXEL = 0x000000009
    CLOCK_PWM   = 0x00000000a
    
    # Voltage IDs
    VOLTAGE_CORE    = 0x000000001
    VOLTAGE_SDRAM_C = 0x000000002
    VOLTAGE_SDRAM_P = 0x000000003
    VOLTAGE_SDRAM_I = 0x000000004
    
    # VC memory flags
    MEM_FLAG_DISCARDABLE      = 1 << 0
    MEM_FLAG_NORMAL           = 0 << 2  # What does it mean?
    MEM_FLAG_DIRECT           = 1 << 2
    MEM_FLAG_COHERENT         = 1 << 3
    MEM_FLAG_L1_NONALLOCATING = (MEM_FLAG_DIRECT | MEM_FLAG_COHERENT)
    MEM_FLAG_ZERO             = 1 << 4
    MEM_FLAG_NO_INIT          = 1 << 5
    MEM_FLAG_HINT_PERMALOCK   = 1 << 6

    def __init__(self):
        self.fd = os.open('/dev/vcio', os.O_RDONLY)

    def close(self):
        if self.fd:
            os.close(self.fd)
        self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return exc_value is None

    def _simple_call(self, name, tag, req_fmt, res_fmt, args):
        'Call a method which has constant length response.'

        # Since the mailbox property interface overwrites the request tag buffer for returning
        # values to the host, size of the buffer must have enough space for both request
        # arguments and returned values. It must also be 32-bit aligned.
        tag_size = (max(calcsize(req_fmt), calcsize(res_fmt)) + 3) // 4 * 4

        buf = array('B', [0]*IOCTL_BUFSIZE)
        pack_into('=5L' + req_fmt + 'L', buf, 0,
                *([24 + tag_size, PROCESS_REQUEST, tag, tag_size, tag_size] + args + [0]))

        ioctl(self.fd, IOCTL_MAILBOX, buf, True)

        r = unpack_from('=5L' + res_fmt, buf, 0)
        if r[1] != REQUEST_SUCCESS:
            raise MailBoxException('Request failed', name, *args)

        assert(r[4] == 0x80000000 | calcsize(res_fmt))
        return r

    @classmethod
    def _add_simple_method(cls, name, tag, req_fmt, res_fmt):
        def f(self, *args):
            r = self._simple_call(name, tag, req_fmt, res_fmt, list(args))[5:]
            n = len(r)
            if n == 1:
                return r[0]
            elif n > 1:
                return r
        setattr(cls, name, f)

    def get_clocks(self):
        buf = array('B', [0]*IOCTL_BUFSIZE)
        pack_into('=5L', buf, 0,
            *[IOCTL_BUFSIZE, PROCESS_REQUEST, 0x00010007, IOCTL_BUFSIZE-24, IOCTL_BUFSIZE-24]
            )

        ioctl(self.fd, IOCTL_MAILBOX, buf, True)

        r = unpack_from('=5L' + res_fmt, buf, 0)
        if r[1] != REQUEST_SUCCESS:
            raise MailBoxException('Request failed', 'get_clocks')
        assert(r[4] & 0x80000000)
        size = (r[4] & ~0x80000000)/4
        return unpack_from('=%dL'%size, buf, 20)

    def get_command_line(self):
        buf = array('B', [0]*IOCTL_BUFSIZE)
        pack_into('=5L', buf, 0,
            *[IOCTL_BUFSIZE, PROCESS_REQUEST, 0x00050001, IOCTL_BUFSIZE-24, IOCTL_BUFSIZE-24]
            )

        ioctl(self.fd, IOCTL_MAILBOX, buf, True)

        r = unpack_from('=5L' + res_fmt, buf, 0)
        if r[1] != REQUEST_SUCCESS:
            raise MailBoxException('Request failed', 'get_command_line')
        assert(r[4] & 0x80000000)
        size = (r[4] & ~0x80000000)
        return unpack_from('=%ds'%size, buf, 20)[0]

    def _palette_method(self, name, tag, offset, length, values):
        buf = array('L',
                [IOCTL_BUFSIZE, PROCESS_REQUEST, tag, (length+2)*4, offset, length] +
                values)
        ioctl(self.fd, IOCTL_MAILBOX, buf, True)
        r = unpack_from('=6L' + res_fmt, buf, 0)
        if r[1] != REQUEST_SUCCESS:
            raise MailBoxException('Request failed', name)
        assert(r[4] == 0x80000004)
        return r[5]

    def test_palette(self, offset, length, values):
        return self._palette_method('test_palette', 0x0004400b, offset, length, values)

    def set_palette(self, offset, length, values):
        return self._palette_method('set_palette', 0x0004800b, offset, length, values)

MAILBOX_METHODS = [
    ('get_firmware_revision',            0x00000001,  '',     'L'),    
    ('get_board_model',                  0x00010001,  '',     'L'),    
    ('get_board_revision',               0x00010002,  '',     'L'),    
    ('get_mac_address',                  0x00010003,  '',     '6s'),   
    ('get_board_serial',                 0x00010004,  '',     'Q'),    
    ('get_arm_memory',                   0x00010005,  '',     'LL'),   
    ('get_vc_memory',                    0x00010006,  '',     'LL'),   
    ('get_power_state',                  0x00020001,  'L',    'LL'),     
    ('get_timing',                       0x00020002,  'L',    'LL'),     
    ('set_power_state',                  0x00028001,  'LL',   'LL'),     
    ('get_clock_state',                  0x00030001,  'L',    'LL'),     
    ('set_clock_state',                  0x00038001,  'LL',   'LL'),     
    ('get_clock_rate',                   0x00030002,  'L',    'LL'),     
    ('set_clock_rate',                   0x00038002,  'LLL',  'LL'),     
    ('get_max_clock_rate',               0x00030004,  'L',    'LL'),     
    ('get_min_clock_rate',               0x00030007,  'L',    'LL'),     
    ('get_turbo',                        0x00030009,  'L',    'LL'),     
    ('set_turbo',                        0x00038009,  'LL',   'LL'),     
    ('get_voltage',                      0x00030003,  'L',    'LL'),     
    ('set_voltage',                      0x00038003,  'LL',   'LL'),     
    ('get_max_voltage',                  0x00030005,  'L',    'LL'),     
    ('get_min_voltage',                  0x00030008,  'L',    'LL'),     
    ('get_temperature',                  0x00030006,  'L',    'LL'),     
    ('get_max_temperature',              0x0003000a,  'L',    'LL'),     
    ('allocate_memory',                  0x0003000c,  'LLL',  'L'),     
    ('lock_memory',                      0x0003000d,  'L',    'L'),     
    ('unlock_memory',                    0x0003000e,  'L',    'L'),     
    ('release_memory',                   0x0003000f,  'L',    'L'),     
    ('execute_code',                     0x00030010,  '7L',   'L'),     
    ('execute_qpu',                      0x00030011,  'LLLL', 'L'),     
    ('enable_qpu',                       0x00030012,  'L',    'L'),     
    ('get_dispmax_resource_mem_handle',  0x00030014,  'L',    'LL'),     
    ('get_edid_block',                   0x00030020,  'L',    'LL128s'),     
    ('allocate_buffer',                  0x00040001,  'L',    'LL'),     
    ('release_buffer',                   0x00048001,  '',     ''),     
    ('blank_screen',                     0x00040002,  'L',    'L'),     
    ('get_physical_display_size',        0x00040003,  '',     'LL'),     
    ('test_physical_display_size',       0x00044003,  'LL',   'LL'),     
    ('set_physical_display_size',        0x00048003,  'LL',   'LL'),     
    ('get_virtual_buffer_size',          0x00040004,  '',     'LL'),     
    ('test_virtual_buffer_size',         0x00044004,  'LL',   'LL'),     
    ('set_virtual_buffer_size',          0x00048004,  'LL',   'LL'),     
    ('get_depth',                        0x00040005,  '',     'L'),     
    ('test_depth',                       0x00044005,  'L',    'L'),     
    ('set_depth',                        0x00048005,  'L',    'L'),     
    ('get_pixel_order',                  0x00040006,  '',     'L'),     
    ('test_pixel_order',                 0x00044006,  'L',    'L'),     
    ('set_pixel_order',                  0x00048006,  'L',    'L'),     
    ('get_alpha_mode',                   0x00040007,  '',     'L'),     
    ('test_alpha_mode',                  0x00044007,  'L',    'L'),     
    ('set_alpha_mode',                   0x00048007,  'L',    'L'),     
    ('get_pitch',                        0x00040008,  '',     'L'),     
    ('get_virtual_offset',               0x00040009,  '',     'LL'),     
    ('test_virtual_offset',              0x00044009,  'LL',   'LL'),     
    ('set_virtual_offset',               0x00048009,  'LL',   'LL'),     
    ('get_overscan',                     0x0004000a,  '',     '4L'),     
    ('test_overscan',                    0x0004400a,  '4L',   '4L'),     
    ('set_overscan',                     0x0004800a,  '4L',   '4L'),     
    ('get_palette',                      0x0004000b,  '',     '1024s'),     
    ('get_dma_channels',                 0x00060001,  '',     ''),     
    ('set_cursor_state',                 0x00008010,  '4L',   'L'),     
    ('set_cursor_info',                  0x00008011,  '6L',   'L'),     
    ]

for name, tag, req_fmt, res_fmt in MAILBOX_METHODS:
    MailBox._add_simple_method(name, tag, req_fmt, res_fmt)
