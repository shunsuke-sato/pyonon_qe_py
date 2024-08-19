import numpy as np
import shlex
#from scipy.stats import qmc
#import scipy
from scipy.stats import qmc


def read_dynamical_matrix(filename):
    """
    Read the dynamical matrix from a Quantum Espresso output file.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()



    # Identify the lines containing the dynamical matrix
    start_idx = None
    for i, line in enumerate(lines):
        if 'Dynamical matrix' in line:
            start_idx = i + 2  # Assuming matrix starts two lines after this header
            break

    if start_idx is None:
        raise ValueError("Dynamical matrix not found in the file.")

    # Read the matrix dimensions
    nelements = int(lines[start_idx].split()[0])
    natoms = int(lines[start_idx].split()[1])
    ibrav = int(lines[start_idx].split()[2])
    lattice_const = float(lines[start_idx].split()[3])

    # Calculate lattice vectors
    a_vec = np.zeros((3,3))
    if(ibrav == 1):
        a_vec[0,0] = lattice_const
        a_vec[1,1] = lattice_const
        a_vec[2,2] = lattice_const
    elif(ibrav == 2):
        a_vec[0,0] = -1.0*lattice_const/2
        a_vec[1,0] =  0.0
        a_vec[2,0] =  1.0*lattice_const/2
        a_vec[0,1] =  0.0
        a_vec[1,1] =  1.0*lattice_const/2
        a_vec[2,1] =  1.0*lattice_const/2
        a_vec[0,2] = -1.0*lattice_const/2
        a_vec[1,2] =  1.0*lattice_const/2
        a_vec[2,2] =  0.0*lattice_const/2
    else:
        raise ValueError("Invalid value")


    # Read the element information
    mass_elem = np.zeros(nelements)
    icount = start_idx
    for i in range(nelements):
        icount += 1
        mass_elem[i] = float(shlex.split(lines[icount])[2])

    # Read the atom information
    atom2elem = np.zeros(natoms, dtype=int)

    atom_alat_coor = np.zeros((natoms, 3))
    for i in range(natoms):
        icount += 1
        atom2elem[i] = int(lines[icount].split()[1])
        atom_alat_coor[i, 0:3] = lines[icount].split()[2:5]

    atom_alat_coor = atom_alat_coor*lattice_const

    # Convert from Cartecian to Reduced coordinates
    atom_red_coor = np.zeros((natoms, 3))
    b_vec = np.linalg.inv(a_vec)
    vec_in = np.zeros(3)
    vec_out = np.zeros(3)
    
    for iatom in range(natoms):
        vec_in[0:3] = atom_alat_coor[iatom,0:3]
        vec_out = np.matmul(b_vec, vec_in)
        atom_red_coor[iatom,0:3] = vec_out[0:3]

    matrix_size = 3 * natoms
        
    # Initialize the dynamical matrix
    dynamical_matrix = np.zeros((matrix_size, matrix_size), dtype=complex)


    icount += 5
    # Read the dynamical matrix elements
    for iatom in range(natoms):
        for jatom in range(natoms):
            icount += 1
            # start from zero
            ielem = atom2elem[iatom]-1
            jelem = atom2elem[jatom]-1
            
            for i in range(3):
                icount += 1
                elements = lines[icount].split()
                for j in range(3):
                    real_part = float(elements[2*j])
                    imag_part = float(elements[2*j+1])
                    dynamical_matrix[3*iatom+i,3*jatom+j] \
                        = (real_part + 1j*imag_part) \
                          /np.sqrt(mass_elem[ielem]*mass_elem[jelem])


    # convert to atomic units (from Ry to Hartree)
    dynamical_matrix = dynamical_matrix/4
    mass_elem = mass_elem*2

    return dynamical_matrix, nelements, natoms, mass_elem, atom2elem, atom_red_coor, a_vec

def diagonalize_dynamical_matrix(dynamical_matrix):
    """
    Diagonalize the dynamical matrix to obtain phonon frequencies and modes.
    """
    # Diagonalize the matrix
    eigenvalues, eigenvectors = np.linalg.eigh(dynamical_matrix)

    # The phonon frequencies are the square roots of the eigenvalues
    phonon_frequencies = np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues)

    return phonon_frequencies, eigenvectors

def wigner_distribution(phonon_frequencies,phonon_modes,nelements,natoms,mass_elem,atom2elem,atom_red_coor,a_vec):
    

    m_num = 4
    num_file = 2**m_num
    kbt = 0.0009500449526217702 # 300 K
    beta = 1.0/kbt
    sampler = qmc.Sobol(d=3)
    sample = sampler.random_base2(m_num)

    for ifile in range(num_file):

        drion_sec = np.zeros(3*natoms)
        xrand_vec = np.random.normal(loc=0.0, scale=1.0, size=3*natoms)
        xrand_vec[0:3] = 0.0

        for imode in range(0+3,3*natoms):
            sigma = 2.0*phonon_frequencies[imode]*np.tanh(beta*phonon_frequencies[imode]/2.0)
            sigma = 1.0/np.sqrt(sigma)
            xrand_vec[imode] = xrand_vec[imode]*sigma


        vec_t = np.matmul(phonon_modes, xrand_vec)
        icount = -1
        for iatom in range(natoms):
            ielem = atom2elem[iatom]-1
            mass = mass_elem[ielem]
            for i in range(3):
                icount += 1
                vec_t[icount] = vec_t[icount]/np.sqrt(mass)

        drion_sec = np.real(vec_t)

        drion = np.zeros((natoms,3))
        icount = -1
        for iatom in range(natoms):
            for i in range(3):
                icount += 1
                drion[iatom,i] = drion_sec[icount]


        # Convert from Cartecian to Reduced coordinates
        b_vec = np.linalg.inv(a_vec)
        vec_t = np.zeros(3)
        atom_red_coor_mod = np.zeros((natoms,3))
        for iatom in range(natoms):
            vec_t[0:3] = drion[iatom,0:3]
            vec_r = np.matmul(b_vec, vec_t)
            atom_red_coor_mod[iatom,0:3] = atom_red_coor[iatom,0:3] + vec_r[0:3]


        # Create files
        nk_t = 1
        while (nk_t**3*natoms < 2*8**3):
            nk_t=nk_t+1


        filename = 'atom_red_coor_wigner_'+str(ifile).zfill(5)+'.out'
        with open(filename, mode='w') as f:
            f.write('&kgrid\n')
            out_str = "num_kgrid(1:3)= "+str(nk_t)+', '+str(nk_t)+', '+str(nk_t)+'\n'
            f.write(out_str)
            csample0 = str(sample[ifile,0]-0.5)
            csample1 = str(sample[ifile,1]-0.5)
            csample2 = str(sample[ifile,2]-0.5)

            out_str='dk_shift(1:3)= '+csample0+', '+csample1+', '+csample2+' \n'
            f.write(out_str)
            f.write('/\n')
            f.write('&atomic_red_coor\n')
            for iatom in range(natoms):
                csample0 = str(atom_red_coor_mod[iatom,0])+'  '
                csample1 = str(atom_red_coor_mod[iatom,1])+'  '
                csample2 = str(atom_red_coor_mod[iatom,2])+'  '
                out_str = "'Si'  "+csample0+csample1+csample2+'  1\n'
                f.write(out_str)
            f.write('/\n')

    return 0

def main():
    # Replace 'dynamical_matrix_file.txt' with the path to your dynamical matrix file
    filename = 'si.dyn'
    
    # Read the dynamical matrix
    dynamical_matrix, nelements, natoms, mass_elem, atom2elem, atom_red_coor, a_vec = read_dynamical_matrix(filename)
    
    # Diagonalize the dynamical matrix to get phonon frequencies and modes
    phonon_frequencies, phonon_modes = diagonalize_dynamical_matrix(dynamical_matrix)

    # Provide Wigner distribution

    a = wigner_distribution(phonon_frequencies,phonon_modes,nelements,natoms,mass_elem,atom2elem,atom_red_coor, a_vec)
    
    # Print the results

    print("Phonon frequencies (in atomic units):")
    print(phonon_frequencies)

    print("Phonon frequencies (eV):")
    print(phonon_frequencies*27.2114)

    print("Phonon frequencies (THz):")
    print(phonon_frequencies*241.8*27.2114)
#    print("\nPhonon modes:")
#    print(phonon_modes)

if __name__ == "__main__":
    main()
