import torch


class _SqrtPositivePart(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ret = torch.sqrt(torch.clamp(x, min=0))
        ctx.save_for_backward(ret)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        ret, = ctx.saved_tensors
        # Derivative of sqrt(x) is 1/(2*sqrt(x))
        # Use a small epsilon to avoid division by zero
        grad_input = grad_output / (2 * ret + 1e-12)
        return grad_input


def safe_rot_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Gradient-safe conversion from rotation matrix to quaternion (w, x, y, z).
    Avoids NaN gradients at the identity matrix caused by eigh.
    """
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    
    def _copysign(a, b):
        signs_differ = (a < 0) != (b < 0)
        return torch.where(signs_differ, -a, a)

    q_abs = torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1)
    
    q_abs = _SqrtPositivePart.apply(q_abs) * 0.5
    
    q_w = q_abs[..., 0]
    q_x = _copysign(q_abs[..., 1], matrix[..., 2, 1] - matrix[..., 1, 2])
    q_y = _copysign(q_abs[..., 2], matrix[..., 0, 2] - matrix[..., 2, 0])
    q_z = _copysign(q_abs[..., 3], matrix[..., 1, 0] - matrix[..., 0, 1])
    
    return torch.stack([q_w, q_x, q_y, q_z], dim=-1)


def conjugate_quat(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)

def multiply_quat(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:

    return torch.cat(
        (
            q2[..., :1] * q1[..., 0:1] - q2[..., 1:2] * q1[..., 1:2] - q2[..., 2:3] * q1[..., 2:3] - q2[..., 3:4] * q1[..., 3:4],
            q2[..., :1] * q1[..., 1:2] + q2[..., 1:2] * q1[..., 0:1] - q2[..., 2:3] * q1[..., 3:4] + q2[..., 3:4] * q1[..., 2:3],
            q2[..., :1] * q1[..., 2:3] + q2[..., 1:2] * q1[..., 3:4] + q2[..., 2:3] * q1[..., 0:1] - q2[..., 3:4] * q1[..., 1:2],
            q2[..., :1] * q1[..., 3:4] - q2[..., 1:2] * q1[..., 2:3] + q2[..., 2:3] * q1[..., 1:2] + q2[..., 3:4] * q1[..., 0:1],
        ),
        dim=-1,
    )

def rotate_vec_by_quat(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    r = v.new_zeros(list(v.shape[:-1]) + [4])
    r[..., 1:] = v

    q_conj = conjugate_quat(q)
    r = multiply_quat(multiply_quat(q, r), q_conj)

    return r[..., 1:]
