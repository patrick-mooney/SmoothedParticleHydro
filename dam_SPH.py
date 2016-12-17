# Patrick Mooney - Math Modeling Project
# Dam break problem using smoothed particle hydrodynamics

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axis as ax
import matplotlib.animation as animation
import matplotlib.cm as cm
import time
from math import sqrt, pi

# Use Monaghan correction, Wendland localization, 
# predictor-corrector numerical integration scheme
# with artificial viscosity and repulsive boundary force

################### Physical Constants ###################
water_density = 997 # kg / m^3
water_viscosity = 1.002e-3 # kg / (m*sec)
kinematic_viscosity = 1.005e-6 # m^2 / sec
v_sound = 1.5e3 # m / sec
atm_pressure = 1e5 # kg / (m*sec^2)
gravity = 9.8 # m / sec^2
g = np.array([0.,-gravity])
##########################################################


################### Simulation Constants ###################
num_parcels = 16 # Number of real parcels
t_steps = 300  # Number of time steps in the simulation - with t_steps = 9000 & h = 0.0005 the simulation will cover 4.5 seconds
h = 0.005      # Time discretization 

reservior_length = int(25./54.*num_parcels**(0.5)) # 25x25 m square block of water is 54x54 parcels
vertical_wall_height = 40                          # Height of left and right boundaries
river_length = 100                                 # Length of river bottom boundary

# Equation of state variables
gamma = 7
mach_number = 0.1
bulk_density = 1000.
bulk_velocity = mach_number * v_sound
B = 10**4 * bulk_velocity**2


sr = 0.4 # Smoothing radius, radius of support
mc = 0.5 # Monaghan correction constant


# Artificial viscosity constants
visc_a = 0.01
visc_b = 0.

# Repulsive boundary force parameters
K = 1000.
delta = 1.2 # Value of (distance from fictitious parcel/initial_parcel_distance) at which the boundary force begins

parcel_mass = water_density * reservior_length**2 / num_parcels
#initial_parcel_distance = reservior_length / num_parcels
initial_parcel_distance = 20./54.
############################################################


# Plot the parcels
def plot_dam(coordinates, t_step, num_parcels, river_length, vertical_wall_height):
	plt.figure(num=2, figsize=(15,6))
	plt.axis([-1, river_length+1, -1, vertical_wall_height])
	plt.scatter(coordinates[:,0], coordinates[:,1], c='black', marker='o')
	plt.xlabel('X (m)')
	plt.ylabel('Y (m)')
	plt.title('SPH Dam Break  t_step= %d  parcel_num = %d' % (t_step, num_parcels))
	plt.show()
	plt.close()


# Wendland localization function
def W_wendland(s, sr):
	if s < sr:
		wnc = 7. / (np.pi * sr**2)
		return wnc * (1 - (s/sr))**4 * ((4*s/sr) + 1)
	return 0.


# Derivative of the Wendland localization function with respect to s
# Used in momentum_balance and density_update
def W_prime_wendland(s, sr):
	if s < sr:
		wnc = 7. / (np.pi * sr**2)
		return wnc * ( ((-4/sr)*(1-(s/sr))**3 * ((4*s/sr) + 1))  +  (4/sr)*(1-(s/sr))**4 )
	return 0.


# Before I had debugged the code significantly, I was using the Gaussian localizations
# They were, however, a source of uncertainty during debugging due to the lambda parameter
# Most likely they work fine, but I switched to the Wendland localization so that I could
# worry about one less thing. They are not used in the simulation currently

# Gaussian localization function
def W_gauss(s, sr):
	lamb = 10.
	gnc = lamb**2 * (np.pi * sr**2)**(-1)
	return gnc * np.exp(-(lamb*s / sr)**2)


# Derivative of the Gaussian localization function with respect to s
# Used in momentum_balance and density update
def W_prime_gauss(s, sr):
	lamb = 10.
	gnc = lamb**2 * (np.pi * sr**2)**(-1)
	return gnc * ( (-2 * lamb**2 * s / sr**2) * np.exp(-(lamb*s / sr)**2))


# Artificial viscosity used in momentum_balance
# xi, xj, ui, uj are 2d vectors
def art_visc(xi, xj, ui, uj, ci, cj, roi, roj, sr, a , b):
	dx = xi - xj
	du = ui - uj
	dot_xu = np.dot(dx, du)
	#print(dot_xu)
	if dot_xu <= 0:
		cij = (ci + cj)/2.
		roij = (roi + roj)/2.
		muij = ( 2*sr*dot_xu ) / ( np.linalg.norm(dx)**2 + (0.2*sr)**2 )
		#print( ( -a*cij*muij + b*muij**2 ) / roij )
		return ( -a*cij*muij + b*muij**2 ) / roij
	
	return 0.


# Monoghan correction to the velocity
# We use this correction to smooth out the velocity changes
# The correction employs the Wendland localization scheme
def mon_cor(i, coordinates, velocities, densities, co_indx, mc, mass, sr, neighbors=[]):
	correction = 0.
	# For now, check whether the neighbor list is implemented
	# Didn't get around to neighbor lists, but that's what the check is for
	if not neighbors:
		for j in range(coordinates.size//2):
			if i == j:
				continue
			s = np.linalg.norm(coordinates[i,:] - coordinates[j,:])
			if s < sr:
				correction += (velocities[i,co_indx] - velocities[j,co_indx]) * W_wendland(s, sr) \
				/ (densities[i] + densities[j])

		correction *= 2. * mc * mass
	return velocities[i,co_indx] - correction


# Used to update velocities using the space derivative of the Wendland localization, and artificial viscosity
# additional environmental forces in this computation are gravity and a repulsion force from the boundary
def  momentum_balance(i, co, u, dens, co_indx, mass, gravity, pr, c, sr, visc_a , visc_b, K, delta, d0, num_real_parcels, neighbors=[]):
	sum = 0.
	boundary_force = 0.
	if not neighbors:
		for j in range(co.size//2):
			if i == j:
				continue
			s = np.linalg.norm(co[i,:] - co[j,:])
			if s < sr:
				artificial_viscosity = art_visc(co[i,:], co[j,:], u[i,:], u[j,:], c[i], c[j], dens[i], dens[j], sr, visc_a , visc_b)

				sum += (W_prime_wendland(s, sr) / s) * ((pr[i]/dens[i]**2) + (pr[j]/dens[j]**2) + artificial_viscosity) \
					* (co[i, co_indx] - co[j,co_indx])

			# Boundary force
			if j >= num_real_parcels and (s/d0) <= delta:
				boundary_force += K * ( (s/d0)**(-0.5) * (delta - (s/d0))**2 * (1./(s*d0)) * (co[i,co_indx] - co[j,co_indx]) )


	return gravity[co_indx] + (boundary_force/mass) - sum * mass


# For particle i, update the density. Notice that even though this is 
# the DE for density, no density term appears in this formulation
def density_update(i, co, u, mass, sr):
	sum = 0.
	for j in range(co.size//2):
		if i == j:
			continue
		s = np.linalg.norm(coordinates[i,:] - coordinates[j,:])
		if s < sr:
			dx = co[i,:] - co[j,:]
			du = u[i,:] - u[j,:]
			dot_xu = np.dot(dx, du)
			sum += (W_prime_wendland(s, sr) / s) * dot_xu
	return mass * sum


# Update speeds of sound in individual parcels
# Sound speed and density will dictate compressibility
def sound_speed_update(c, dens, bulk_density, B, gamma):
	for i in range(c.size):
		c[i] = ( (B * gamma) / bulk_density )**(0.5) * (dens[i] / bulk_density)**((gamma-1)/2.)
	return c


# Update the pressures of real parcels using the equation of state
def pressure_update(press, dens, bulk_density, B, gamma):
	for i in range(press.size):
		press[i] = B * ( (dens[i] / bulk_density)**gamma - 1)
	return press




################### Initializations ###################

# First, initialize the positions of each parcel into a square
coordinates = np.zeros((num_parcels,2))
for i in range(int(sqrt(num_parcels))): # i is the vertical distance
	for j in range(int(sqrt(num_parcels))): # j is the horizontal distance
		coordinates[i*int(sqrt(num_parcels))+j,0] = (j+1)*initial_parcel_distance  # j+1 so the parcel isn't in boundary
		coordinates[i*int(sqrt(num_parcels))+j,1] = (i+1)*initial_parcel_distance 


# Initialize fictitious parcels along the walls and floor
wall_fparcels_y = np.array([initial_parcel_distance*i/2. for i in range(int(vertical_wall_height/(initial_parcel_distance/2.)))])
floor_fparcels_x = np.array([initial_parcel_distance*i/2. for i in range(int(river_length/(initial_parcel_distance/2.)))])
left_wall_fparcels = np.zeros((len(wall_fparcels_y),2))
left_wall_fparcels[:,1] = np.copy(wall_fparcels_y)
right_wall_fparcels = np.copy(left_wall_fparcels)
right_wall_fparcels[:,0] = river_length
floor_fparcels = np.zeros((len(floor_fparcels_x),2))
floor_fparcels[:,0] = np.copy(floor_fparcels_x)
fparcel_coordinates = np.concatenate((left_wall_fparcels, floor_fparcels, right_wall_fparcels),axis=0)
num_fparcels = fparcel_coordinates.size // 2

# After the index [num_parcels-1], coordinates holds the positions of fictitious parcels
coordinates = np.concatenate((coordinates, fparcel_coordinates), axis = 0)


# Initialize velocities to zero for real parcels, and perpendicular to the boundary
# for the fictitious parcels. The corner velocities point diagonally out
real_vel = np.zeros((num_parcels, 2))
fictitious_u = 1.0
left_wall_u = np.copy(left_wall_fparcels)
left_wall_u[:,0] = fictitious_u
left_wall_u[:,1] = 0.
right_wall_u = np.copy(left_wall_fparcels)
right_wall_u[:,0] = -fictitious_u
right_wall_u[:,1] = 0.
floor_u = np.copy(floor_fparcels)
floor_u[:,0] = 0.
floor_u[:,1] = fictitious_u

p_vel = np.concatenate((real_vel, left_wall_u, right_wall_u, floor_u), axis=0)


# Initialize densities using the equation of state
# Watch out for initializing fictitious densities with y > wd
p_dens = np.zeros(num_parcels + num_fparcels)
for i in range(p_dens.size):
	p_dens[i] = bulk_density * ( (bulk_density * gravity * (reservior_length - coordinates[i,1]) / B ) + 1)**(1./gamma)


# Initialize pressures using the equation of state
# They will be calculated the same way during the simulation
p_press = np.zeros(num_parcels + num_fparcels)
for i in range(p_press.size):
	p_press[i] = B * ( (p_dens[i] / bulk_density)**gamma - 1)


# Initialize sound speeds
p_vsound = np.zeros(num_parcels + num_fparcels)
for i in range(p_vsound.size):
	p_vsound[i] = (B * gamma / bulk_density)**(0.5) * (p_dens[i] / bulk_density)**((gamma-1)/2.)


# Initialize storage for predictor-corrector and improved euler
half_step_u = np.copy( p_vel[:,:] ) # holds data for all parcels real and fictitious so that it can be used in functions
half_step_x = np.copy( coordinates[:,:] )
half_step_d = np.copy( p_dens[:] )
old_velocity_halves = np.copy( p_vel[:num_parcels,:] )
old_position_halves = np.copy( coordinates[:num_parcels,:] )
old_density_halves = np.copy( p_dens[:num_parcels] )
euler_velocity_func = np.copy(old_velocity_halves)
euler_position_func = np.copy(old_position_halves)
euler_density_func = np.copy(old_density_halves)


# Initialize storage for new values of diff eqs
new_densities = np.zeros(num_parcels)
new_velocities = np.zeros((num_parcels,2))
new_coordinates = np.zeros((num_parcels,2))


# Show initial setup
# plot_dam(coordinates, t_steps, num_parcels, river_length, vertical_wall_height)


# Set up for animation
# Use initial_pr variables to scale pressure colormap in animation
initial_pr_bottom = p_press[num_parcels] # pressure of bottom parcel in boundary
initial_pr_top = p_press[(num_parcels-1)+int(vertical_wall_height/(initial_parcel_distance/2.))] # pressure of top parcel on boundary
fig = plt.figure('animation', figsize=(15,6))
img = []

#######################################################




# Begin simulation
###################
start_time = time.time()


# INTEGRATION STARTUP ROUTINE
# Use improved Euler to do the first time step by stepping twice by h/2
for n in range(2):
	# NOTE: This first inner for loop below is why we are doing this startup step at all. It is all to get these old_---_halves 
	# to be used in the predictor corrector. In particular, the second (and last) iteration of the outer loop is what will give us
	# the values we need to continue with the bulk of the simulation.
	for i in range(num_parcels):
		old_velocity_halves[i,0] = momentum_balance(i, half_step_x, half_step_u, half_step_d, 0, parcel_mass, g, p_press,\
		 												 p_vsound, sr, visc_a, visc_b, K, delta, initial_parcel_distance, num_parcels)
		old_velocity_halves[i,1] = momentum_balance(i, half_step_x, half_step_u, half_step_d, 1, parcel_mass, g, p_press,\
		 												 p_vsound, sr, visc_a, visc_b, K, delta, initial_parcel_distance, num_parcels)
		old_position_halves[i,0] = mon_cor(i, half_step_x, half_step_u, half_step_d, 0, mc, parcel_mass, sr)
		old_position_halves[i,1] = mon_cor(i, half_step_x, half_step_u, half_step_d, 1, mc, parcel_mass, sr)
		old_density_halves[i] = density_update(i, half_step_x, half_step_u, parcel_mass, sr)

	for i in range(num_parcels):
		half_step_u[i,0] = p_vel[i,0] + (h/2.) * old_velocity_halves[i,0]
		half_step_u[i,1] = p_vel[i,1] + (h/2.) * old_velocity_halves[i,1]
		half_step_x[i,0] = coordinates[i,0] + (h/2.) * old_position_halves[i,0]
		half_step_x[i,1] = coordinates[i,1] + (h/2.) * old_position_halves[i,1]
		half_step_d[i] = p_dens[i] + (h/2.) * old_density_halves[i]

	p_press[:] = pressure_update(p_press, half_step_d, bulk_density, B, gamma)
	p_vsound[:] = sound_speed_update(p_vsound, half_step_d, bulk_density, B, gamma)

	for i in range(num_parcels):
		euler_velocity_func[i,0] = momentum_balance(i, half_step_x, half_step_u, half_step_d, 0, parcel_mass, g, p_press,\
		 												 p_vsound, sr, visc_a, visc_b, K, delta, initial_parcel_distance, num_parcels)
		euler_velocity_func[i,1] = momentum_balance(i, half_step_x, half_step_u, half_step_d, 1, parcel_mass, g, p_press,\
		 												 p_vsound, sr, visc_a, visc_b, K, delta, initial_parcel_distance, num_parcels)
		euler_position_func[i,0] = mon_cor(i, half_step_x, half_step_u, half_step_d, 0, mc, parcel_mass, sr)
		euler_position_func[i,1] = mon_cor(i, half_step_x, half_step_u, half_step_d, 1, mc, parcel_mass, sr)
		euler_density_func[i] = density_update(i, half_step_x, half_step_u, parcel_mass, sr)

	p_vel[:num_parcels,:] += (h/4.) * (old_velocity_halves + euler_velocity_func)
	coordinates[:num_parcels,:] += (h/4.) * (old_position_halves + euler_position_func)
	p_dens[:num_parcels] += (h/4.) * (old_density_halves + euler_density_func)
	p_press[:] = pressure_update(p_press, p_dens, bulk_density, B, gamma)      # Don't actually need to update these now for next time step
	p_vsound[:] = sound_speed_update(p_vsound, p_dens, bulk_density, B, gamma) # but do so to display the proper data



# Continue simulation for subsequent states using predictor-corrector method
for n in range(t_steps):
	# Update neighbor lists here, if implemented
	
	# Update half steps
	for i in range(num_parcels):
		# Velocity half step
		half_step_u[i,0] = p_vel[i,0] + (h/2.) * old_velocity_halves[i,0]
		half_step_u[i,1] = p_vel[i,1] + (h/2.) * old_velocity_halves[i,1]

		# Position half step
		half_step_x[i,0] = coordinates[i,0] + (h/2.) * old_position_halves[i,0]
		half_step_x[i,1] = coordinates[i,1] + (h/2.) * old_position_halves[i,1]

		# Density half step
		half_step_d[i] = p_dens[i] + (h/2.) * old_density_halves[i]


	# Update pressures and sound speeds for half step function evaluation
	p_press[:] = pressure_update(p_press, half_step_d, bulk_density, B, gamma)
	p_vsound[:] = sound_speed_update(p_vsound, half_step_d, bulk_density, B, gamma)


	# Update old function halves
	for i in range(num_parcels):
		old_velocity_halves[i,0] = momentum_balance(i, half_step_x, half_step_u, half_step_d, 0, parcel_mass, g, p_press,\
	 												 p_vsound, sr, visc_a, visc_b, K, delta, initial_parcel_distance, num_parcels)
		old_velocity_halves[i,1] = momentum_balance(i, half_step_x, half_step_u, half_step_d, 1, parcel_mass, g, p_press,\
	 												 p_vsound, sr, visc_a, visc_b, K, delta, initial_parcel_distance, num_parcels)

		old_position_halves[i,0] = mon_cor(i, half_step_x, half_step_u, half_step_d, 0, mc, parcel_mass, sr)
		old_position_halves[i,1] = mon_cor(i, half_step_x, half_step_u, half_step_d, 1, mc, parcel_mass, sr)

		old_density_halves[i] = density_update(i, half_step_x, half_step_u, parcel_mass, sr)


	# Calculate and store values for the next time step
	p_vel[:num_parcels,:] += h * old_velocity_halves
	coordinates[:num_parcels,:] += h * old_position_halves
	p_dens[:num_parcels] += h * old_density_halves
	p_press[:] = pressure_update(p_press, p_dens, bulk_density, B, gamma)      # Don't actually need to update these now for next time step
	p_vsound[:] = sound_speed_update(p_vsound, p_dens, bulk_density, B, gamma) # but do so to display the proper data


	# Every 5 iterations, take a snapshot of the parcel positions to be used for the animation
	if n % 2 == 0:
		img.append([plt.scatter(coordinates[:,0], coordinates[:,1], c=p_press[:], cmap=cm.magma, \
								vmin=initial_pr_top, vmax=initial_pr_bottom, marker='o')])

	if n % 50 == 0:	
		print('\n')
		print(n)
		print('{0:50s} {1:50s}'.format(str(coordinates[0,:]), str(coordinates[num_parcels-1,:])))
		print('{0:50s} {1:50s}'.format(str(p_vel[0,:]), str(p_vel[num_parcels-1,:])))
		print('{0:50s} {1:50s}'.format(str(p_dens[0]), str(p_dens[num_parcels-1])))
		print('{0:50s} {1:50s}'.format(str(p_press[0]), str(p_press[num_parcels-1])))
		print('{0:50s} {1:50s}'.format(str(p_vsound[0]), str(p_vsound[num_parcels-1])))

		farthest_right = 0
		for i in range(1, num_parcels):
			if coordinates[i,0] > coordinates[farthest_right,0]:
				farthest_right = i
		print('Horizontal velocity of right-most parcel: %f' % (p_vel[farthest_right, 0]))



# Display duration of simulation
end_time = time.time() - start_time
print('\nTime: %f \n' % (end_time))


# Display the animation
anim = animation.ArtistAnimation(fig, img, interval =50, blit=True)
plt.show()
plt.close()