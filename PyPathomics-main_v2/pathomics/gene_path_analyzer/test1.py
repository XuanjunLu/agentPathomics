def demo():
    currentTerm = None
    # with open(filePath, 'r', encoding='utf-8') as file:
    with open('go-basic.obo', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith('[Term]'):
                if currentTerm:
                    return currentTerm
                currentTerm = {}
            elif line.startswith('id:'):
                currentTerm['id'] = line.split('id: ')[1]
            elif line.startswith('name:'):
                currentTerm['name'] = line.split('name: ')[1]
            elif line.startswith('is_a:'):
                parentId = line.split('is_a: ')[1].split(' ! ')[0]
                if 'is_a' not in currentTerm:
                    currentTerm['is_a'] = []
                currentTerm['is_a'].append(parentId)
    if currentTerm:
        return currentTerm


if __name__ == '__main__':
    demo()