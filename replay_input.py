from pygbx import Gbx, GbxType
from pygbx.headers import ControlEntry, CGameCtnGhost
from numpy import int32

def get_event_time(event: ControlEntry) -> int:
    return (event.time - 1) // 10 * 10

def should_skip_event(event: ControlEntry):
    if event.event_name in ['AccelerateReal', 'BrakeReal']:
        return event.flags != 1

    if event.event_name == 'Steer':
        return False

    if event.event_name.startswith('_Fake'):
        return True

    return False


def event_to_analog_value(event: ControlEntry):
    val = int32((event.flags << 16) | event.enabled)
    val <<= int32(8)
    val >>= int32(8)
    return -val


def try_parse_old_ghost(g: Gbx):
    ghost = CGameCtnGhost(0)

    parser = g.find_raw_chunk_id(0x2401B00F)
    if parser:
        ghost.login = parser.read_string()

    parser = g.find_raw_chunk_id(0x2401B011)
    if parser:
        parser.seen_loopback = True
        g.read_ghost_events(ghost, parser, 0x2401B011)
        return ghost

    return None


def get_inputs(ghost: CGameCtnGhost):
    invert_axis = False
    for event in ghost.control_entries:
        if event.event_name == '_FakeDontInverseAxis':
            invert_axis = True

    events = filter(lambda e: not should_skip_event(e), ghost.control_entries)
    events = [(get_event_time(event),event) for event in events]
    return events
    # for i, event in enumerate(ghost.control_entries):
    #     if should_skip_event(event):
    #         continue
    #
    #     _from = get_event_time(event)
    #
    #     # Always throw out the millisecond precision
    #     _from = int(_from / 10) * 10
    #     _to = int(_to / 10) * 10
    #
    #     action = 'press'
    #     key = 'up'
    #
    #     if event.event_name == 'Accelerate' or event.event_name == 'AccelerateReal':
    #         key = 'up'
    #     elif event.event_name == 'SteerLeft':
    #         key = 'left'
    #     elif event.event_name == 'SteerRight':
    #         key = 'right'
    #     elif event.event_name == 'Brake' or event.event_name == 'BrakeReal':
    #         key = 'down'
    #     elif event.event_name == 'Respawn':
    #         key = 'enter'
    #     elif event.event_name == 'Steer':
    #         action = 'steer'
    #         axis = event_to_analog_value(event)
    #         if invert_axis:
    #             axis = -axis
    #         write_func(f'{_from} {action} {axis}\n')
    #         continue
    #     elif event.event_name == 'Gas':
    #         action = 'gas'
    #         axis = event_to_analog_value(event)
    #         if invert_axis:
    #             axis = -axis
    #
    #         write_func(f'{_from} {action} {axis}\n')
    #         continue
    #     elif event.event_name == 'Horn':
    #         continue
    #
    #     if is_unbound:
    #         write_func(f'{_from} {action} {key}\n')
    #     else:
    #         write_func(f'{_from}-{_to} {action} {key}\n')

def get_inputs_from_replay(path, ghost_index = 1):
    g = Gbx(path)

    ghosts = g.get_classes_by_ids([GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])
    if not ghosts:
        ghost = try_parse_old_ghost(g)
    else:
        ghost = ghosts[ghost_index]

    assert ghost is not None
    return get_inputs(ghost)


def get_positions_from_replay(path, ghost_index = 1):
    g = Gbx(path)

    ghosts = g.get_classes_by_ids([GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])
    if not ghosts:
        ghost = try_parse_old_ghost(g)
    else:
        ghost = ghosts[ghost_index]

    assert ghost is not None
    return ghost.sample_period, [[r.position.x, r.position.y, r.position.z] for r in ghost.records]