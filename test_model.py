"""
单元测试：环境、模型与决策模块
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from environment import (
    ACTION_DIM,
    ACTION_HIGH,
    ACTION_LOW,
    STATE_DIM,
    GameState,
    StrikeAction,
    compute_optimal_action,
    generate_random_state,
    normalize_state,
)
from model import StrikeNet, build_model
from train import generate_dataset, train


# ── GameState ─────────────────────────────────────────────────────────────────

class TestGameState:
    def test_to_vector_shape(self):
        state = GameState([1, 2, 3], [0.5, -1, 0], [0, 0, 1])
        vec = state.to_vector()
        assert vec.shape == (STATE_DIM,)

    def test_to_vector_values(self):
        state = GameState([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0])
        vec = state.to_vector()
        np.testing.assert_array_equal(vec, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_dtype_is_float32(self):
        state = GameState([1, 2, 3], [4, 5, 6], [7, 8, 9])
        assert state.to_vector().dtype == np.float32


# ── StrikeAction ──────────────────────────────────────────────────────────────

class TestStrikeAction:
    def test_to_vector_shape(self):
        action = StrikeAction(0.1, 0.5, 15.0)
        assert action.to_vector().shape == (ACTION_DIM,)

    def test_to_velocity_shape(self):
        action = StrikeAction(0.0, np.pi / 6, 10.0)
        vel = action.to_velocity()
        assert vel.shape == (3,)

    def test_zero_horizontal_angle_no_x_velocity(self):
        # 水平角为 0 时，vx 应为 0
        action = StrikeAction(0.0, np.pi / 6, 10.0)
        vel = action.to_velocity()
        assert abs(vel[0]) < 1e-5


# ── normalize_state ───────────────────────────────────────────────────────────

class TestNormalizeState:
    def test_output_range(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            state = generate_random_state(rng)
            vec = normalize_state(state.to_vector())
            # generate_random_state 的采样范围与归一化边界一致，
            # 因此输出应严格在 [-1, 1]（加 1e-4 容许浮点误差）
            assert np.all(vec >= -1.0 - 1e-4) and np.all(vec <= 1.0 + 1e-4), \
                f"归一化结果超出预期范围: {vec}"

    def test_output_shape(self):
        state = generate_random_state()
        vec = normalize_state(state.to_vector())
        assert vec.shape == (STATE_DIM,)


# ── compute_optimal_action ────────────────────────────────────────────────────

class TestComputeOptimalAction:
    def test_output_shape(self):
        state = generate_random_state()
        action = compute_optimal_action(state)
        assert action.shape == (ACTION_DIM,)

    def test_within_action_bounds(self):
        rng = np.random.default_rng(42)
        for _ in range(200):
            state = generate_random_state(rng)
            action = compute_optimal_action(state)
            assert np.all(action >= ACTION_LOW - 1e-5), f"动作低于下界: {action}"
            assert np.all(action <= ACTION_HIGH + 1e-5), f"动作超出上界: {action}"


# ── generate_random_state ─────────────────────────────────────────────────────

class TestGenerateRandomState:
    def test_returns_game_state(self):
        state = generate_random_state()
        assert isinstance(state, GameState)

    def test_reproducible_with_seed(self):
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        s1 = generate_random_state(rng1)
        s2 = generate_random_state(rng2)
        np.testing.assert_array_equal(s1.to_vector(), s2.to_vector())


# ── StrikeNet / build_model ───────────────────────────────────────────────────

class TestStrikeNet:
    def test_forward_output_shape(self):
        model = build_model()
        x = torch.randn(8, STATE_DIM)
        out = model(x)
        assert out.shape == (8, ACTION_DIM)

    def test_output_within_action_bounds(self):
        model = build_model()
        model.eval()
        x = torch.randn(64, STATE_DIM)
        with torch.no_grad():
            out = model(x).numpy()
        low = ACTION_LOW
        high = ACTION_HIGH
        assert np.all(out >= low - 1e-4), "模型输出低于动作下界"
        assert np.all(out <= high + 1e-4), "模型输出超出动作上界"

    def test_custom_hidden_sizes(self):
        model = build_model(hidden_sizes=(64, 64, 32))
        x = torch.randn(4, STATE_DIM)
        out = model(x)
        assert out.shape == (4, ACTION_DIM)

    def test_save_and_load(self):
        model = build_model()
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        try:
            torch.save(model.state_dict(), path)
            loaded = build_model()
            loaded.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )
            x = torch.randn(4, STATE_DIM)
            model.eval()
            loaded.eval()
            with torch.no_grad():
                np.testing.assert_array_almost_equal(
                    model(x).numpy(), loaded(x).numpy()
                )
        finally:
            os.unlink(path)


# ── generate_dataset ──────────────────────────────────────────────────────────

class TestGenerateDataset:
    def test_shapes(self):
        states, actions = generate_dataset(100, seed=0)
        assert states.shape == (100, STATE_DIM)
        assert actions.shape == (100, ACTION_DIM)

    def test_reproducible(self):
        s1, a1 = generate_dataset(50, seed=7)
        s2, a2 = generate_dataset(50, seed=7)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(a1, a2)


# ── train ─────────────────────────────────────────────────────────────────────

class TestTrain:
    def test_loss_decreases(self):
        losses = train(
            num_samples=500,
            epochs=20,
            batch_size=64,
            lr=1e-3,
            save_path=os.path.join(tempfile.gettempdir(), "test_strike_model.pth"),
            verbose=False,
        )
        assert len(losses) == 20
        # 损失应整体下降：末尾 5 轮均值 < 前 5 轮均值
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, \
            f"损失未下降: 前5轮均值={early_avg:.6f}, 末5轮均值={late_avg:.6f}"

    def test_model_file_created(self):
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        os.unlink(path)  # 先删除，测试训练后是否会创建
        try:
            train(
                num_samples=200,
                epochs=2,
                batch_size=64,
                save_path=path,
                verbose=False,
            )
            assert os.path.isfile(path)
        finally:
            if os.path.isfile(path):
                os.unlink(path)


# ── StrikeDecisionMaker ───────────────────────────────────────────────────────

class TestStrikeDecisionMaker:
    """集成测试：训练后使用 StrikeDecisionMaker 进行推理"""

    @pytest.fixture(scope="class")
    def model_path(self, tmp_path_factory):
        path = str(tmp_path_factory.mktemp("models") / "dm_model.pth")
        train(
            num_samples=1000,
            epochs=5,
            batch_size=64,
            save_path=path,
            verbose=False,
        )
        return path

    def test_decide_returns_strike_action(self, model_path):
        from decision import StrikeDecisionMaker

        dm = StrikeDecisionMaker(model_path=model_path)
        state = GameState([2.0, 1.0, 2.5], [-3.0, 4.0, -1.5], [-4.0, 0.5, 1.0])
        action = dm.decide(state)
        assert isinstance(action, StrikeAction)

    def test_decide_action_within_bounds(self, model_path):
        from decision import StrikeDecisionMaker

        dm = StrikeDecisionMaker(model_path=model_path)
        rng = np.random.default_rng(0)
        for _ in range(20):
            state = generate_random_state(rng)
            action = dm.decide(state)
            vec = action.to_vector()
            assert np.all(vec >= ACTION_LOW - 1e-4)
            assert np.all(vec <= ACTION_HIGH + 1e-4)

    def test_decide_batch(self, model_path):
        from decision import StrikeDecisionMaker

        dm = StrikeDecisionMaker(model_path=model_path)
        rng = np.random.default_rng(1)
        states = [generate_random_state(rng) for _ in range(8)]
        actions = dm.decide_batch(states)
        assert len(actions) == 8
        for a in actions:
            assert isinstance(a, StrikeAction)

    def test_missing_model_raises(self):
        from decision import StrikeDecisionMaker

        with pytest.raises(FileNotFoundError):
            StrikeDecisionMaker(model_path="/nonexistent/model.pth")
