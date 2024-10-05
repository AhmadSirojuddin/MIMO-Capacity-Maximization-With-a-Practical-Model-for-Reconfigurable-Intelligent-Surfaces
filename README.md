This code relates to my article titled *MIMO Capacity Maximization With a Practical Model for Reconfigurable Intelligent Surfaces* published in IEEE Wireless Communications Letters.
The journal can be accessed via this link: https://ieeexplore.ieee.org/document/10681131
The pre-peer-reviewer version of the journal can be accessed at https://www.researchgate.net/publication/384189324_MIMO_Capacity_Maximization_with_a_Practical_Model_for_Reconfigurable_Intelligent_Surfaces

The code is written in Python with the `numpy` framework for linear algebra computation, along with other frameworks like `matplotlib` for plotting, `time` for measuring computation time, etc.

The following files generate the plots that appear in the paper:
- `plot_SE_time_vs_M.py` -> Fig. 2 and 4
- `plot_SE_vs_Resist_v2.py` -> Fig. 3

Those two files call the cores functions:
- `MIMO_RIS_Prac.py` -> the proposed algorithm
- `Judd22.py` -> my previous work based on ideally-modeled RIS. Please refer to https://github.com/AhmadSirojuddin/Low-Complexity-Sum-Capacity-Maximization-for-Intelligent-Reflecting-Surface-Aided-MIMO-Systems 
- `Boyu20.py` -> one benchmark paper
- `Wang21.py` -> one benchmark paper
Disclaimer: the last two codes are self-remaking. If you want the authors' original code, please email them directly.

I am open to research collaborations. Please contact me at: sirojuddin@its.ac.id 
