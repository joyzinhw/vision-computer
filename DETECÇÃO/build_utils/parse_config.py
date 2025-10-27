def parse_model_cfg(path):
    """
    Lê um arquivo de configuração YOLO (.cfg) e retorna uma lista de dicionários
    representando cada camada.
    """
    module_defs = []

    with open(path, 'r') as file:
        lines = file.read().split('\n')
        lines = [x.strip() for x in lines if x and not x.startswith('#')]

    module_def = {}
    for line in lines:
        if line.startswith('['):  # nova seção
            if module_def:
                module_defs.append(module_def)
            module_def = {'type': line[1:-1].strip()}
        else:
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()
            # tenta converter para int ou float, se possível
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
            module_def[key] = value
    module_defs.append(module_def)
    return module_defs
