



```
ros2 launch ur_simulation_gz ur_sim_control.launch.py
```

## Joint Trahectory Controller

```
ros2 run ur5e_admittance_controller example_move.py
```



## Cartesian Velocity Contoller


```
ros2 run ur5e_admittance_controller cartesian_velocity_controller.py

```


```
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/command_cart_vel

```

```
ros2 topic pub /command_cart_vel geometry_msgs/msg/Twist "linear:
  x: 0.1
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0"
```
### References

1. [Compliant Control and Application](https://github.com/MingshanHe/Compliant-Control-and-Application)