# ARX LIFT2s custom branch

This branch is the integrated working branch for the delivered LIFT2s system.
The official `main` branch remains an unchanged comparison point.

## Tracked customizations

- `realsense/realsense.sh` contains the verified head, left-wrist, and
  right-wrist D405 serial-number mapping.
- `custom_sdk/LIFT/body/ROS2` vendors only the delivered body component that
  this branch modifies. Unmodified CAN, arm, and VR components continue to use
  the delivered sibling directory at `/home/arx/LIFT`.
- The body controller exposes `/lift_height_lock` (`std_srvs/srv/SetBool`).
  Locking freezes the most recent lift-height command from VR. Explicit
  `/body_control` commands remain authoritative and update the locked target.
- `act/collect.py` enables the lift-height lock before recording. Collection is
  refused if the lock service is unavailable. The lock intentionally remains
  enabled after collection; unlocking must be an explicit operator action.
- Collection and inference launchers use the body controller tracked by this
  branch.

## Height workflow

1. Start the custom body controller only from a safe low position.
2. With the lock disabled, set the desired height before collection.
3. Start `collect.py`; it locks the current height command before waiting for
   the episode start gesture.
4. Do not restart the body controller while the lift is raised.
5. To adjust height later, explicitly unlock it with:

   ```bash
   ros2 service call /lift_height_lock std_srvs/srv/SetBool "{data: false}"
   ```

## Branch policy

- Keep only `main` (official) and `acceptance/official-chain` (integrated
  working branch) in this repository.
- Put future ARX customizations in this branch and add only the modified SDK
  component under `custom_sdk`.
- Commit each independently testable behavior change; never edit `main`.
