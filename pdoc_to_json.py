#!/usr/bin/env python3
import sys
import os
import json
import argparse
from pathlib import Path
from pdoc import extract, doc

"""
이 스크립트는 패키지 구조 트리(패키지→모듈→클래스→메서드/변수)만 추출하여
최소화된 JSON으로 저장합니다.
"""

def extract_members(mod_obj, prefix):
    """
    모듈 또는 클래스의 멤버(클래스, 함수, 변수)만 추출합니다.
    prefix 기준으로 프로젝트 내부 멤버만 포함하고, dunder 멤버는 제외합니다.
    """
    nodes = []
    if hasattr(mod_obj, 'members'):
        for child in (mod_obj.members.values() if isinstance(mod_obj.members, dict) else mod_obj.members):
            # 프로젝트 내부 멤버만
            if not child.fullname.startswith(prefix + '.'):
                continue
            # magic/dunder 이름 제외
            if child.name.startswith('__') and child.name.endswith('__'):
                continue
            if child.kind == 'class':
                # 클래스 노드
                class_node = {'name': child.name, 'children': []}
                # 클래스 내부 멤버(메서드·변수)
                if hasattr(child, 'members'):
                    for sub in (child.members.values() if isinstance(child.members, dict) else child.members):
                        if sub.name.startswith('__') and sub.name.endswith('__'):
                            continue
                        class_node['children'].append({'name': sub.name, 'children': []})
                nodes.append(class_node)
            elif child.kind in ('function', 'variable'):
                nodes.append({'name': child.name, 'children': []})
    return nodes


def build_tree(module_names, prefix):
    """
    모듈 이름 리스트를 받아, 중첩된 dict 구조로 패키지 트리를 만듭니다.
    각 모듈의 멤버는 '_members' 키로 저장됩니다.
    """
    tree_map = {}
    for full in module_names:
        # prefix로 시작하지 않는 모듈 스킵
        if not full.startswith(prefix + '.') and full != prefix:
            continue
        parts = full.split('.')
        curr = tree_map
        for part in parts:
            curr = curr.setdefault(part, {})
        # 멤버 추출
        try:
            mod_obj = doc.Module.from_name(full)
            curr['_members'] = extract_members(mod_obj, prefix)
        except Exception:
            curr['_members'] = []
    return tree_map


def map_to_nodes(name, subtree):
    """
    트리 맵(dict)에서 JSON 노드 형식으로 변환합니다.
    'name' 및 'children'만 유지합니다.
    """
    children = []
    # 먼저 모듈 내 멤버
    for mem in subtree.get('_members', []):
        children.append(mem)
    # 그 다음 하위 모듈 재귀
    for key, val in subtree.items():
        if key == '_members':
            continue
        children.append(map_to_nodes(key, val))
    return {'name': name, 'children': children}


def main():
    parser = argparse.ArgumentParser(
        description='패키지 모듈 클래스 메서드/변수 트리만 JSON으로 추출합니다.'
    )
    parser.add_argument('package', help='패키지 디렉터리 또는 모듈 경로 (예: ./src)')
    parser.add_argument('-o', '--output', default='tree.json', help='생성할 JSON 파일명 (기본: tree.json)')
    args = parser.parse_args()

    pkg_path = Path(args.package)
    if not pkg_path.exists():
        print(f"Error: '{pkg_path}' does not exist.")
        sys.exit(1)

    prefix = pkg_path.name
    # 프로젝트 루트 import 경로 추가
    sys.path.insert(0, os.getcwd())

    # 모듈 탐색
    all_modules = extract.walk_specs([pkg_path])
    module_names = [n for n in all_modules if '__pycache__' not in n]

    # 트리 맵 생성
    tree_map = build_tree(module_names, prefix)
    # 루트 노드 생성
    root_subtree = tree_map.get(prefix, {})
    tree = map_to_nodes(prefix, root_subtree)

    # 결과 디렉터리 생성
    out_root = Path('pdoc_json_output')
    out_root.mkdir(parents=True, exist_ok=True)
    out_file = out_root / args.output

    # 미니파이 JSON으로 저장
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(tree, f, separators=(',', ':'), ensure_ascii=False)

    print(f"✅ Saved minimal tree JSON: {out_file}")

if __name__ == '__main__':
    main()
