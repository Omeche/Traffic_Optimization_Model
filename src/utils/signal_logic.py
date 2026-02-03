"""
Traffic signal control logic and timing calculations
"""

import numpy as np


class TrafficSignalController:
    """Traffic signal controller with timing logic"""
    
    def __init__(self):
        """Initialize signal controller"""
        self.yellow_time = 3  # seconds
        self.all_red_time = 2  # seconds
        self.min_green = 7  # seconds
        self.max_green = 60  # seconds
        
    def validate_timings(self, cycle_time, green_times):
        """
        Validate signal timings
        
        Args:
            cycle_time: Total cycle time
            green_times: List of green times for each phase
            
        Returns:
            bool, str: (is_valid, message)
        """
        # Check green time constraints
        for i, green in enumerate(green_times):
            if green < self.min_green:
                return False, f"Phase {i+1} green time ({green}s) below minimum ({self.min_green}s)"
            if green > self.max_green:
                return False, f"Phase {i+1} green time ({green}s) above maximum ({self.max_green}s)"
        
        # Check if timings fit within cycle
        num_phases = len(green_times)
        total_lost_time = num_phases * (self.yellow_time + self.all_red_time)
        total_green = sum(green_times)
        required_cycle = total_green + total_lost_time
        
        if required_cycle > cycle_time:
            return False, f"Total time ({required_cycle}s) exceeds cycle time ({cycle_time}s)"
        
        return True, "Timings valid"
    
    def calculate_delay(self, arrival_rate, cycle_time, green_time):
        """
        Calculate average vehicle delay using Webster's delay formula
        
        Args:
            arrival_rate: Vehicle arrival rate (veh/s)
            cycle_time: Signal cycle time (s)
            green_time: Green time for phase (s)
            
        Returns:
            Average delay per vehicle (s)
        """
        if green_time <= 0 or arrival_rate <= 0:
            return 0
        
        # Calculate degree of saturation
        capacity = green_time / cycle_time  # Effective green ratio
        x = arrival_rate * cycle_time / green_time
        
        # Prevent division by zero
        if x >= 0.95:
            x = 0.95
        
        # Webster's delay formula (simplified)
        uniform_delay = (cycle_time * (1 - capacity)**2) / (2 * (1 - x * capacity))
        
        # Add random delay component
        random_delay = x**2 / (2 * arrival_rate * (1 - x))
        
        total_delay = uniform_delay + random_delay
        
        # Cap unreasonable values
        return min(total_delay, 300)  # Max 5 minutes
    
    def optimize_phase_sequence(self, flows, current_sequence):
        """
        Optimize phase sequence to minimize conflicts
        
        Args:
            flows: Dictionary of traffic flows by movement
            current_sequence: Current phase sequence
            
        Returns:
            Optimized phase sequence
        """
        # Simple optimization: prioritize higher flow phases
        # In practice, this would consider more complex conflict matrices
        
        phase_priorities = []
        for i, phase in enumerate(current_sequence):
            # Assume each phase has associated flows
            total_flow = sum(flows.get(f"phase_{i}_flow", [0]))
            phase_priorities.append((phase, total_flow))
        
        # Sort by flow (descending)
        optimized = sorted(phase_priorities, key=lambda x: x[1], reverse=True)
        
        return [phase for phase, _ in optimized]
    
    def calculate_pedestrian_clearance(self, crossing_distance, walking_speed=1.2):
        """
        Calculate pedestrian clearance time
        
        Args:
            crossing_distance: Distance to cross (m)
            walking_speed: Walking speed (m/s)
            
        Returns:
            Clearance time (s)
        """
        # Time to cross
        crossing_time = crossing_distance / walking_speed
        
        # Add startup time (time for pedestrian to start crossing)
        startup_time = 3
        
        # Add safety buffer
        safety_buffer = 2
        
        total_time = crossing_time + startup_time + safety_buffer
        
        return np.ceil(total_time)
    
    def adjust_for_priority(self, base_green, priority_vehicle_present):
        """
        Adjust green time for priority vehicles (buses, emergency)
        
        Args:
            base_green: Base green time (s)
            priority_vehicle_present: Boolean indicating priority vehicle
            
        Returns:
            Adjusted green time (s)
        """
        if priority_vehicle_present:
            # Extend green time by 10 seconds for priority
            adjusted = base_green + 10
        else:
            adjusted = base_green
        
        # Ensure within bounds
        return np.clip(adjusted, self.min_green, self.max_green)
    
    def calculate_cycle_split(self, cycle_time, phase_ratios):
        """
        Calculate green time splits for each phase
        
        Args:
            cycle_time: Total cycle time (s)
            phase_ratios: List of ratios for each phase (should sum to 1)
            
        Returns:
            List of green times
        """
        num_phases = len(phase_ratios)
        
        # Calculate available green time
        total_lost_time = num_phases * (self.yellow_time + self.all_red_time)
        available_green = cycle_time - total_lost_time
        
        # Distribute according to ratios
        green_times = []
        for ratio in phase_ratios:
            green = available_green * ratio
            # Ensure minimum
            green = max(green, self.min_green)
            green_times.append(green)
        
        # Normalize if total exceeds available
        total_allocated = sum(green_times)
        if total_allocated > available_green:
            scale_factor = available_green / total_allocated
            green_times = [g * scale_factor for g in green_times]
        
        return green_times


class AdaptiveSignalControl:
    """Adaptive signal control strategies"""
    
    def __init__(self):
        """Initialize adaptive controller"""
        self.history_window = 5  # Number of past cycles to consider
        self.adjustment_rate = 0.1  # How quickly to adapt
        
    def actuated_control(self, detector_occupancy, base_green, max_extension=20):
        """
        Actuated control: extend green based on detector occupancy
        
        Args:
            detector_occupancy: Percentage of detector occupied (0-100)
            base_green: Base green time
            max_extension: Maximum extension allowed
            
        Returns:
            Extended green time
        """
        # Extend if occupancy is high
        if detector_occupancy > 80:
            extension = max_extension
        elif detector_occupancy > 50:
            extension = max_extension * 0.5
        else:
            extension = 0
        
        return base_green + extension
    
    def gap_out_control(self, vehicle_gaps, critical_gap=3.0, base_green=20):
        """
        Gap-out control: terminate green when gap exceeds threshold
        
        Args:
            vehicle_gaps: List of time gaps between vehicles (s)
            critical_gap: Critical gap threshold (s)
            base_green: Base green time
            
        Returns:
            Actual green time served
        """
        elapsed_green = 0
        
        for gap in vehicle_gaps:
            if gap > critical_gap and elapsed_green >= base_green:
                # Gap out - terminate green
                break
            elapsed_green += gap
        
        return elapsed_green
    
    def max_pressure_control(self, incoming_queue, outgoing_space):
        """
        Max-pressure control: prioritize phases with maximum pressure
        
        Args:
            incoming_queue: Number of vehicles in incoming lanes
            outgoing_space: Available space in outgoing lanes
            
        Returns:
            Pressure value (higher = more priority)
        """
        # Pressure = demand - supply
        pressure = incoming_queue - outgoing_space
        
        return max(0, pressure)
    
    def adaptive_cycle_length(self, traffic_volume, base_cycle=90):
        """
        Adapt cycle length based on traffic volume
        
        Args:
            traffic_volume: Current traffic volume
            base_cycle: Base cycle length
            
        Returns:
            Adjusted cycle length
        """
        # Increase cycle for high volume, decrease for low
        if traffic_volume > 100:  # High volume
            adjustment = 1.2
        elif traffic_volume > 50:  # Medium volume
            adjustment = 1.0
        else:  # Low volume
            adjustment = 0.8
        
        adjusted_cycle = base_cycle * adjustment
        
        # Constrain to reasonable bounds
        return np.clip(adjusted_cycle, 60, 150)


def calculate_level_of_service(delay):
    """
    Calculate Level of Service (LOS) based on delay
    
    Args:
        delay: Average delay per vehicle (s)
        
    Returns:
        LOS grade (A-F)
    """
    if delay <= 10:
        return 'A'
    elif delay <= 20:
        return 'B'
    elif delay <= 35:
        return 'C'
    elif delay <= 55:
        return 'D'
    elif delay <= 80:
        return 'E'
    else:
        return 'F'


def estimate_throughput(green_time, cycle_time, saturation_flow=1800):
    """
    Estimate intersection throughput
    
    Args:
        green_time: Green time per cycle (s)
        cycle_time: Total cycle time (s)
        saturation_flow: Saturation flow rate (veh/hour)
        
    Returns:
        Throughput (veh/hour)
    """
    # Capacity = saturation flow * (green / cycle)
    capacity = saturation_flow * (green_time / cycle_time)
    
    return capacity


def calculate_queue_dissipation_time(queue_length, saturation_flow=1800):
    """
    Calculate time to clear a queue
    
    Args:
        queue_length: Number of vehicles in queue
        saturation_flow: Flow rate during green (veh/hour)
        
    Returns:
        Time to clear queue (s)
    """
    # Convert saturation flow to veh/s
    flow_per_second = saturation_flow / 3600
    
    # Time to clear
    dissipation_time = queue_length / flow_per_second
    
    return dissipation_time