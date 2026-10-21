class AppError(Exception):
    """Error previsible de uso (archivo, columnas, tipos, versión, modelo...).

    Los scripts lo muestran como un mensaje claro (sin traza) y terminan con
    código de salida distinto de cero.
    """
