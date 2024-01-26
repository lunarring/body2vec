import torch

def uniform_slerp(vectors, alpha):
    n = vectors.size(0)
    
    # Calculate segment size and find the relevant vectors for interpolation
    segment_size = 1.0 / n
    index = int(alpha/segment_size)

    alpha_normalized = (alpha - index * segment_size) / segment_size

    # print(f"{alpha} indices are {index} and {(index + 1) % n}")

    v0 = vectors[index]
    v1 = vectors[(index + 1) % n]

    # Spherical interpolation
    dot = torch.dot(v0, v1)
    dot = torch.clamp(dot, -1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    if sin_theta > 0.001:
        a = torch.sin((1.0 - alpha_normalized) * theta) / sin_theta
        b = torch.sin(alpha_normalized * theta) / sin_theta
        return a * v0 + b * v1
    else:
        # If the angle is small, linear interpolation is a good approximation
        return (1.0 - alpha_normalized) * v0 + alpha_normalized * v1



if __name__=="__main__":
    # Example usage
    vectors = torch.tensor([
        [1.0,0.0,0.0,0.0],
        [0.0,1.0,0.0,0.0],
        [0.0,0.0,1.0,0.0],
    ])  # n vectors on the hypersphere
    alpha = 7/8  # Example alpha value

    for i in range(10):
        print(uniform_slerp(vectors, i/10))