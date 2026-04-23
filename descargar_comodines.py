import os
import requests
from bs4 import BeautifulSoup
import re

def limpiar_nombre(nombre):
    # Reemplaza espacios por barra baja y limpia caracteres raros
    nombre_limpio = "".join([c for c in nombre if c.isalnum() or c in (' ', '_', '-')]).strip()
    return nombre_limpio.replace(" ", "_")

def obtener_url_original(url_thumb):
    """
    Transforma la URL de la miniatura de MediaWiki a la URL original.
    Ejemplo thumb: .../images/thumb/a/a1/Joker.png/100px-Joker.png
    Original:      .../images/a/a1/Joker.png
    """
    if not url_thumb: return None
    
    # 1. Asegurar esquema https
    if url_thumb.startswith('//'):
        url_thumb = 'https:' + url_thumb
    elif url_thumb.startswith('/'):
        url_thumb = 'https://balatrowiki.org' + url_thumb

    # 2. Si es una miniatura (/thumb/), intentamos sacar la original
    if '/thumb/' in url_thumb:
        try:
            # La estructura suele ser: /images/thumb/A/AB/Archivo.png/##px-Archivo.png
            # Queremos quedarnos con lo que hay ANTES del último slash (la versión reducida)
            # y quitar la parte de /thumb/
            
            # Paso A: Quitar el último segmento (el de "120px-...")
            base_url = url_thumb.rsplit('/', 1)[0]
            
            # Paso B: Quitar "/thumb" de la ruta
            original_url = base_url.replace('/thumb/', '/')
            
            return original_url
        except:
            return url_thumb # Si falla, devolvemos la original por si acaso
            
    return url_thumb

def descargar_jokers_wiki():
    url = "https://balatrowiki.org/w/Jokers"
    headers = {'User-Agent': 'Mozilla/5.0 (Compatible; BalatroBot/1.0)'}
    
    print(f"Conectando a {url} ...")
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Error al conectar: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Carpeta de destino
    folder = "jokers_balatrowiki"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # En MediaWiki, las tablas suelen tener la clase 'wikitable'
    # Buscamos todas las filas de todas las tablas
    tables = soup.find_all('table', class_='wikitable')
    
    if not tables:
        # Intento alternativo por si no usan la clase wikitable
        tables = soup.find_all('table')
    
    print(f"Se encontraron {len(tables)} tablas. Analizando...")
    
    count = 0
    descargados = set() # Para evitar duplicados

    for table in tables:
        rows = table.find_all('tr')
        
        # Saltamos cabeceras
        for row in rows:
            cols = row.find_all(['td', 'th'])
            
            # Buscamos una fila que tenga al menos imagen y texto
            # Normalmente: [Imagen] [Nombre] [Efecto] ... o [Nombre] [Imagen] ...
            
            img_tag = row.find('img')
            if not img_tag:
                continue
                
            # Extraer nombre: Buscamos el texto en la primera celda que tenga texto y sea negrita o enlace,
            # o simplemente iteramos las celdas.
            nombre_raw = ""
            
            # Estrategia: Buscar el primer enlace (a) que no sea una imagen
            enlaces = row.find_all('a')
            for enlace in enlaces:
                # Si el enlace no tiene una imagen dentro, probablemente es el nombre
                if not enlace.find('img') and enlace.get_text(strip=True):
                    nombre_raw = enlace.get_text(strip=True)
                    break
            
            # Si falló lo anterior, coger el texto de la segunda celda (asumiendo col 0 es imagen)
            if not nombre_raw and len(cols) > 1:
                nombre_raw = cols[1].get_text(strip=True)
                
            # Limpieza final del nombre
            filename = limpiar_nombre(nombre_raw)
            
            if not filename or len(filename) < 2:
                continue
                
            if filename in descargados:
                continue

            # Procesar URL de la imagen
            src = img_tag.get('src')
            url_final = obtener_url_original(src)
            
            try:
                # Descargar
                r_img = requests.get(url_final, headers=headers, timeout=10)
                
                # Si falla la original (a veces pasa), intentamos la thumb original
                if r_img.status_code != 200:
                    r_img = requests.get(obtener_url_original(src), headers=headers) # Retry simple
                    if r_img.status_code != 200:
                        # Fallback: descargar la miniatura tal cual
                        if src.startswith('//'): src = 'https:' + src
                        elif src.startswith('/'): src = 'https://balatrowiki.org' + src
                        r_img = requests.get(src, headers=headers)
                
                r_img.raise_for_status()
                
                # Guardar
                path = os.path.join(folder, f"{filename}.png")
                with open(path, 'wb') as f:
                    f.write(r_img.content)
                
                print(f" [OK] {filename}.png")
                descargados.add(filename)
                count += 1
                
            except Exception as e:
                print(f" [ERROR] {filename}: {e}")

    print(f"\n✅ Proceso completado. {count} imágenes guardadas en '{folder}'.")

if __name__ == "__main__":
    descargar_jokers_wiki()