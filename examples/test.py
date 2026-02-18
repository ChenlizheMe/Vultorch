import vultorch
print('imread:', hasattr(vultorch, 'imread'))
print('imwrite:', hasattr(vultorch, 'imwrite'))
print(vultorch._vultorch._imread.__doc__)