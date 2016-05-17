import pytest 

from lifetimes.generate_data import bgbb_model_decompressed as bgbb_gen

@pytest.mark.generateData
def test_BGBBBB_transaction():
    N = 10
    T = 20
    gen_data = bgbb_gen(T,0.3,5,0.6,10,N)
    assert len(gen_data) == N
    for user in gen_data:
        for t in user[1]:
            assert t != 0
            assert user[0] == T

