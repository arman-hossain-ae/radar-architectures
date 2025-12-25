import numpy as np
import matplotlib.pyplot as plt

class SignalGenerator:
    def __init__(self, sampling_rate, transmit_frequency, analog_oscillator_frequency,
                 transmit_power, gain,
                 pulse_duration, listening_duration,
                 initial_velocity, initial_range, wave_speed,
                 target_reflectivity=0.3, noise=True, system_impedance=50):
        
        self.sampling_rate = sampling_rate
        self.transmit_frequency = transmit_frequency
        self.analog_oscillator_frequency = analog_oscillator_frequency
        self.pulse_duration = pulse_duration
        self.listening_duration = listening_duration
        
        self.velocity = initial_velocity
        self.current_velocity = initial_velocity
        self.range = initial_range
        self.current_range = initial_range
        self.wave_speed = wave_speed
        
        self.nominal_frequency = transmit_frequency - analog_oscillator_frequency
        
        # Noise floor
        self.K = 1.38e-23
        self.T = 413.0
        self.B = 1 / pulse_duration
        self.F = 2
        Pn_watts = self.K * self.T * self.B * self.F
        self.noise_voltage_rms = np.sqrt(Pn_watts * system_impedance)
        
        # Radar equation
        self.pt = transmit_power
        self.g = gain
        self.lambda_ = self.wave_speed / transmit_frequency
        self.sigma = target_reflectivity
        
        # Timing
        self.total_samples = int(listening_duration * sampling_rate)
        self.pulse_samples = int(pulse_duration * sampling_rate)
        self.pulse_no = 0
        self.pri = listening_duration + pulse_duration
        
    def generate(self, target=True, acceleration=0):
        self.update_range(acceleration)
        t = np.arange(0, self.listening_duration, 1/self.sampling_rate)
        
        noise = np.random.normal(0, self.noise_voltage_rms, size=t.shape)
        signal = noise.copy()
        
        if target:
            pr_watts = (self.pt * self.g**2 * self.lambda_**2 * self.sigma) / ((4*np.pi)**3 * self.current_range**4 * self.F)
            echo_voltage_rms = np.sqrt(pr_watts * 50)  # scale with impedance
            
            tau = 2 * self.current_range / self.wave_speed
            delay_samples = int(np.round(tau * self.sampling_rate))
            
            echo_t = t[:self.pulse_samples]
            doppler = 2 * self.current_velocity / self.lambda_
            
            echo = echo_voltage_rms * np.cos(2 * np.pi * (self.nominal_frequency + doppler) * echo_t)
            pulse_samples = min(self.pulse_samples, self.total_samples - delay_samples) 
            signal[delay_samples:delay_samples + pulse_samples] += echo[:pulse_samples]
        
        return signal
    
    def update_range(self, acceleration=0):
        time = self.pulse_no * self.pri
        self.current_range = self.range + self.current_velocity * time + 0.5 * acceleration * time**2
        self.current_velocity += acceleration * self.pri
        self.pulse_no += 1
    
    def reset(self):
        self.current_range = self.range
        self.current_velocity = self.velocity
        self.pulse_no = 0

    
if __name__ == "__main__":
    generator = SignalGenerator(25e6, 10e9, 9.9e9, 2000, 45, 5e-6, 995e-6, 1, 10000000, 3e8)
    generator.generate()